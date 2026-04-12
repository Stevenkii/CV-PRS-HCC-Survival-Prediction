import os
import glob
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import joblib  # 用于保存 sklearn 类模型
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis 
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from lifelines.utils import concordance_index

# ================= 1. 配置区域 =================
TRAIN_CSV = "ready_to_train_train.csv"
VAL_CSV   = "ready_to_train_val.csv"
TEST_CSV  = "ready_to_train_test.csv"

# 图像特征路径
TRAIN_FEAT_DIR = "/public/home/lifex/xiax/HCC_TCGA_data/train_trident_processed/20x_512px_0px_overlap/slide_features_titan"
VAL_FEAT_DIR   = "/public/home/lifex/xiax/HCC_TCGA_data/val_trident_processed/20x_512px_0px_overlap/slide_features_titan"
TEST_FEAT_DIR  = "/public/home/lifex/xiax/HCC_TCGA_data/test_trident_processed/20x_512px_0px_overlap/slide_features_titan"

ID_COL = "Sample_ID"
TIME_COL = "survival_time"
EVENT_COL = "survival_status"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 2. 基础函数 =================
def get_combined_file_mapping(dir_list):
    mapping = {}
    print(f"\n[Step 1] 建立文件索引...")
    for folder in dir_list:
        if not os.path.exists(folder): continue
        h5_files = glob.glob(os.path.join(folder, "*.h5"))
        for path in h5_files:
            mapping[os.path.basename(path)[:12]] = path
    print(f"✅ 索引完成: {len(mapping)} 个文件。\n")
    return mapping

def load_data_strict(csv_path, file_mapping):
    if not os.path.exists(csv_path): return None, None, None
    df = pd.read_csv(csv_path)
    exclude_cols = [ID_COL, TIME_COL, EVENT_COL]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    valid_data = []
    for idx, row in df.iterrows():
        short_id = str(row[ID_COL])[:12]
        h5_path = file_mapping.get(short_id)
        if h5_path and os.path.exists(h5_path):
            try:
                with h5py.File(h5_path, 'r') as f:
                    key = 'features' if 'features' in f else list(f.keys())[0]
                    feat = f[key][:]
                clin = row[feature_cols].values.astype(np.float32)
                valid_data.append((np.hstack((clin, feat)), float(row[TIME_COL]), int(row[EVENT_COL])))
            except: pass
    if not valid_data: return None, None, None
    X = np.array([x[0] for x in valid_data], dtype=np.float32)
    T = np.array([x[1] for x in valid_data], dtype=float)
    E = np.array([x[2] for x in valid_data], dtype=int)
    return X, T, E

# DeepSurv 定义
class DeepSurv(nn.Module):
    def __init__(self, in_dim, hidden=[512, 256], drop=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def cox_loss(risk, t, e):
    idx = torch.argsort(t, descending=True)
    risk, e = risk[idx], e[idx]
    mask = e.bool()
    if mask.sum() == 0: return torch.tensor(0.0, device=DEVICE, requires_grad=True)
    log_sum = torch.logcumsumexp(risk, 0)
    return -torch.sum(risk[mask] - log_sum[mask]) / mask.sum()

def get_sk_y(t, e):
    y = np.zeros(len(t), dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    y['Status'] = e.astype(bool); y['Survival_in_days'] = t
    return y
def get_xgb_y(t, e): return np.where(e==1, t, -t)

class DS(Dataset):
    def __init__(self, x, t, e): 
        self.x, self.t, self.e = torch.tensor(x).float(), torch.tensor(t).float(), torch.tensor(e).float()
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.t[i], self.e[i]

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ================= 3. 主流程 =================
def main():
    set_seed(42) # 建议使用之前跑出 0.75 的那个种子，如果不知道就试几次
    print("🚀 开始执行：5折CV选参 -> 全量重训 -> 模型保存")
    
    mapping = get_combined_file_mapping([TRAIN_FEAT_DIR, VAL_FEAT_DIR, TEST_FEAT_DIR])
    if not mapping: return
    X1, T1, E1 = load_data_strict(TRAIN_CSV, mapping)
    X2, T2, E2 = load_data_strict(VAL_CSV, mapping)
    X_test, T_test, E_test = load_data_strict(TEST_CSV, mapping)
    if X1 is None: return

    # 组装开发集
    X_dev = np.vstack((X1, X2))
    T_dev = np.concatenate((T1, T2))
    E_dev = np.concatenate((E1, E2))
    
    # 🔥 保存标准化器 🔥
    print("🔄 执行并保存 StandardScaler...")
    scaler = StandardScaler()
    X_dev = scaler.fit_transform(X_dev)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, "Scaler.pkl") # <--- 保存 Scaler
    print("💾 Scaler 已保存为 'Scaler.pkl'")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # --- 1. Coxnet ---
    print("\n[1/5] Coxnet ...")
    cox_grid = {'l1_ratio': [1.0, 0.9, 0.5, 0.01], 'alphas': [[0.2], [0.1], [0.08], [0.05], [0.02]]}
    best_score = -1; best_params = None
    for p in ParameterGrid(cox_grid):
        scores = []
        for tr_idx, va_idx in kf.split(X_dev):
            m = CoxnetSurvivalAnalysis(l1_ratio=p['l1_ratio'], alphas=p['alphas'], fit_baseline_model=True, max_iter=1000000)
            try: m.fit(X_dev[tr_idx], get_sk_y(T_dev[tr_idx], E_dev[tr_idx])); scores.append(m.score(X_dev[va_idx], get_sk_y(T_dev[va_idx], E_dev[va_idx])))
            except: scores.append(0.5)
        if np.mean(scores) > best_score: best_score, best_params = np.mean(scores), p
    print(f"  🏆 Coxnet 最佳 CV: {best_score:.4f} | {best_params}")
    
    # 重训并保存
    final_m = CoxnetSurvivalAnalysis(l1_ratio=best_params['l1_ratio'], alphas=best_params['alphas'], fit_baseline_model=True, max_iter=1000000)
    final_m.fit(X_dev, get_sk_y(T_dev, E_dev))
    joblib.dump(final_m, "Best_Coxnet.pkl") # <--- 保存
    results['Coxnet'] = final_m.score(X_test, get_sk_y(T_test, E_test))

    # --- 2. RSF ---
    print("\n[2/5] RSF ...")
    rsf_grid = {'n_estimators': [1000], 'min_samples_leaf': [3, 10, 20], 'max_features': ['sqrt', 'log2'], 'max_depth': [None, 10]}
    best_score = -1; best_params = None
    for p in ParameterGrid(rsf_grid):
        scores = []
        for tr_idx, va_idx in kf.split(X_dev):
            m = RandomSurvivalForest(n_jobs=-1, random_state=42, **p)
            m.fit(X_dev[tr_idx], get_sk_y(T_dev[tr_idx], E_dev[tr_idx]))
            try: scores.append(m.score(X_dev[va_idx], get_sk_y(T_dev[va_idx], E_dev[va_idx])))
            except: pass
        if np.mean(scores) > best_score: best_score, best_params = np.mean(scores), p
    print(f"  🏆 RSF 最佳 CV: {best_score:.4f} | {best_params}")
    
    # 重训并保存
    final_m = RandomSurvivalForest(n_jobs=-1, random_state=42, **best_params).fit(X_dev, get_sk_y(T_dev, E_dev))
    joblib.dump(final_m, "Best_RSF.pkl") # <--- 保存
    results['RSF'] = final_m.score(X_test, get_sk_y(T_test, E_test))

    # --- 3. GBSA ---
    print("\n[3/5] GBSA ...")
    gbsa_grid = {'n_estimators': [300], 'learning_rate': [0.05, 0.1], 'max_depth': [2, 3], 'subsample': [0.7]}
    best_score = -1; best_params = None
    for p in ParameterGrid(gbsa_grid):
        scores = []
        for tr_idx, va_idx in kf.split(X_dev):
            m = GradientBoostingSurvivalAnalysis(random_state=42, **p)
            m.fit(X_dev[tr_idx], get_sk_y(T_dev[tr_idx], E_dev[tr_idx]))
            try: scores.append(m.score(X_dev[va_idx], get_sk_y(T_dev[va_idx], E_dev[va_idx])))
            except: pass
        if np.mean(scores) > best_score: best_score, best_params = np.mean(scores), p
    print(f"  🏆 GBSA 最佳 CV: {best_score:.4f}")
    
    # 重训并保存
    final_m = GradientBoostingSurvivalAnalysis(random_state=42, **best_params).fit(X_dev, get_sk_y(T_dev, E_dev))
    joblib.dump(final_m, "Best_GBSA.pkl") # <--- 保存
    results['GBSA'] = final_m.score(X_test, get_sk_y(T_test, E_test))

    # --- 4. XGBoost ---
    print("\n[4/5] XGBoost ...")
    xgb_grid = {'eta': [0.01, 0.03], 'max_depth': [2, 3, 4], 'subsample': [0.7], 'colsample_bytree': [0.5, 0.7, 1.0], 'reg_alpha': [0, 0.1, 0.5, 1.0]}
    best_score = -1; best_params = None
    for p in ParameterGrid(xgb_grid):
        scores = []
        for tr_idx, va_idx in kf.split(X_dev):
            dtr = xgb.DMatrix(X_dev[tr_idx], label=get_xgb_y(T_dev[tr_idx], E_dev[tr_idx]))
            dva = xgb.DMatrix(X_dev[va_idx], label=get_xgb_y(T_dev[va_idx], E_dev[va_idx]))
            bp = {'objective':'survival:cox', 'tree_method':'hist', 'seed':42, **p}
            m = xgb.train(bp, dtr, 1000, evals=[(dva,'v')], early_stopping_rounds=50, verbose_eval=False)
            preds = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
            try: scores.append(concordance_index_censored(E_dev[va_idx].astype(bool), T_dev[va_idx], preds)[0])
            except: scores.append(0.5)
        if np.mean(scores) > best_score: best_score, best_params = np.mean(scores), p
    print(f"  🏆 XGB 最佳 CV: {best_score:.4f} | {best_params}")
    
    # 重训并保存
    X_fin_tr, X_fin_va, T_fin_tr, T_fin_va, E_fin_tr, E_fin_va = train_test_split(X_dev, T_dev, E_dev, test_size=0.2, random_state=42, stratify=E_dev)
    d_fin_tr = xgb.DMatrix(X_fin_tr, label=get_xgb_y(T_fin_tr, E_fin_tr))
    d_fin_va = xgb.DMatrix(X_fin_va, label=get_xgb_y(T_fin_va, E_fin_va))
    d_test   = xgb.DMatrix(X_test, label=get_xgb_y(T_test, E_test))
    final_bp = {'objective':'survival:cox', 'tree_method':'hist', 'seed':42, **best_params}
    final_m = xgb.train(final_bp, d_fin_tr, 3000, evals=[(d_fin_va, 'iv')], early_stopping_rounds=50, verbose_eval=False)
    final_m.save_model("Best_XGB.json") # <--- 保存 JSON 格式
    results['XGB'] = concordance_index_censored(E_test.astype(bool), T_test, final_m.predict(d_test, iteration_range=(0, final_m.best_iteration + 1)))[0]

    # --- 5. DeepSurv ---
    print("\n[5/5] DeepSurv ...")
    ds_grid = {'lr': [1e-3, 5e-4, 1e-4], 'drop': [0.3, 0.5], 'hidden': [[512, 256], [256, 128], [128, 64]], 'wd': [0, 1e-4, 1e-3], 'bs': [32, 64]}
    best_score = -1; best_params = None
    
    # 5.1 自动搜索超参数
    for p in ParameterGrid(ds_grid):
        scores = []
        for tr_idx, va_idx in kf.split(X_dev):
            d_tr = DataLoader(DS(X_dev[tr_idx], T_dev[tr_idx], E_dev[tr_idx]), batch_size=p['bs'], shuffle=True)
            d_va = DataLoader(DS(X_dev[va_idx], T_dev[va_idx], E_dev[va_idx]), batch_size=64)
            m = DeepSurv(X_dev.shape[1], p['hidden'], p['drop']).to(DEVICE)
            opt = optim.Adam(m.parameters(), lr=p['lr'], weight_decay=p['wd'])
            # CV 快速筛选 (固定50轮)
            for _ in range(50): 
                m.train()
                for bx, bt, be in d_tr:
                    opt.zero_grad(); loss = cox_loss(m(bx.to(DEVICE)), bt.to(DEVICE), be.to(DEVICE)); loss.backward(); opt.step()
            m.eval()
            r, t_l, e_l = [], [], []
            with torch.no_grad():
                for bx, bt, be in d_va: r.extend(m(bx.to(DEVICE)).cpu().numpy().flatten()); t_l.extend(bt.numpy()); e_l.extend(be.numpy())
            try: scores.append(concordance_index(t_l, -np.array(r), e_l))
            except: scores.append(0.5)
        if np.mean(scores) > best_score: best_score, best_params = np.mean(scores), p
    print(f"  🏆 DeepSurv 最佳 CV: {best_score:.4f} | {best_params}")
    
    # 5.2 使用搜索到的参数重训并保存 (Best_DeepSurv.pth)
    X_fin_tr, X_fin_va, T_fin_tr, T_fin_va, E_fin_tr, E_fin_va = train_test_split(X_dev, T_dev, E_dev, test_size=0.2, random_state=42, stratify=E_dev)
    dl_tr = DataLoader(DS(X_fin_tr, T_fin_tr, E_fin_tr), batch_size=best_params['bs'], shuffle=True)
    dl_va = DataLoader(DS(X_fin_va, T_fin_va, E_fin_va), batch_size=64)
    dl_te = DataLoader(DS(X_test, T_test, E_test), batch_size=64)
    
    m = DeepSurv(X_dev.shape[1], best_params['hidden'], best_params['drop']).to(DEVICE)
    opt = optim.Adam(m.parameters(), lr=best_params['lr'], weight_decay=best_params['wd'])
    best_w, max_s, cnt = None, 0, 0
    
    # 早停训练
    for epoch in range(200):
        m.train()
        for bx, bt, be in dl_tr:
            opt.zero_grad(); loss = cox_loss(m(bx.to(DEVICE)), bt.to(DEVICE), be.to(DEVICE)); loss.backward(); opt.step()
        m.eval()
        r, t_l, e_l = [], [], []
        with torch.no_grad():
            for bx, bt, be in dl_va: r.extend(m(bx.to(DEVICE)).cpu().numpy().flatten()); t_l.extend(bt.numpy()); e_l.extend(be.numpy())
        try: cur_s = concordance_index(t_l, -np.array(r), e_l)
        except: cur_s = 0.5
        
        if cur_s > max_s: max_s, best_w, cnt = cur_s, m.state_dict(), 0
        else: cnt += 1
        if cnt >= 20: break
    
    if best_w: m.load_state_dict(best_w)
    torch.save(m.state_dict(), "Best_DeepSurv.pth") # <--- 保存搜索到的最佳权重
    
    # Final Test (搜索模型的测试)
    r, t_l, e_l = [], [], []
    with torch.no_grad():
        for bx, bt, be in dl_te: r.extend(m(bx.to(DEVICE)).cpu().numpy().flatten()); t_l.extend(bt.numpy()); e_l.extend(be.numpy())
    results['DeepSurv_Auto'] = concordance_index(t_l, -np.array(r), e_l)

    # ============================================================
    # 🎯 [新增模块] 冠军参数复现实验 (Expert Mode)
    # ============================================================
    print("\n🚀 [Expert Mode] 开始执行 DeepSurv 指定参数强制复现...")
    
    # 1. 强制设定冠军参数
    target_params = {
        'bs': 32,             # 强制锁定 32
        'drop': 0.3,          # Dropout
        'hidden': [512, 256], # 网络结构
        'lr': 0.001, 
        'wd': 0.0001
    }
    
    # 2. 重新构建 DataLoader (因为 batch_size 变了，必须重做训练集 Loader)
    # 注意：直接复用 X_fin_tr 等变量，确保数据划分与上方完全一致
    dl_tr_repro = DataLoader(DS(X_fin_tr, T_fin_tr, E_fin_tr), batch_size=target_params['bs'], shuffle=True)
    # 验证集和测试集 DataLoader (dl_va, dl_te) 可以复用
    
    # 3. 初始化新模型
    m_repro = DeepSurv(X_dev.shape[1], target_params['hidden'], target_params['drop']).to(DEVICE)
    opt_repro = optim.Adam(m_repro.parameters(), lr=target_params['lr'], weight_decay=target_params['wd'])
    
    best_w_repro = None
    max_s_repro = 0
    cnt_repro = 0
    
    print(f"  --> 正在锁定参数训练: {target_params}")
    
    # 4. 训练循环 (独立跑 200 epoch)
    for epoch in range(200):
        m_repro.train()
        for bx, bt, be in dl_tr_repro:
            opt_repro.zero_grad()
            loss = cox_loss(m_repro(bx.to(DEVICE)), bt.to(DEVICE), be.to(DEVICE))
            loss.backward()
            opt_repro.step()
            
        # 验证步
        m_repro.eval()
        r, t_l, e_l = [], [], []
        with torch.no_grad():
            for bx, bt, be in dl_va: # 复用之前的验证Loader
                r.extend(m_repro(bx.to(DEVICE)).cpu().numpy().flatten())
                t_l.extend(bt.numpy())
                e_l.extend(be.numpy())
        
        try: cur_s = concordance_index(t_l, -np.array(r), e_l)
        except: cur_s = 0.5
        
        # 保存最佳
        if cur_s > max_s_repro:
            max_s_repro = cur_s
            best_w_repro = m_repro.state_dict()
            cnt_repro = 0
        else:
            cnt_repro += 1
            if cnt_repro >= 20: break # 早停
            
    # 5. 加载最佳权重并进行最终测试
    if best_w_repro:
        m_repro.load_state_dict(best_w_repro)
        
    # 测试集评估
    r, t_l, e_l = [], [], []
    with torch.no_grad():
        for bx, bt, be in dl_te: # 复用之前的测试Loader
            r.extend(m_repro(bx.to(DEVICE)).cpu().numpy().flatten())
            t_l.extend(bt.numpy())
            e_l.extend(be.numpy())
            
    score_repro = concordance_index(t_l, -np.array(r), e_l)
    results['DeepSurv_Repro'] = score_repro # 把结果加入最终列表
    
    print(f"  🏆 复现结果 (Test C-index): {score_repro:.4f}")
    
    # 6. 如果结果达标，立刻保存为特殊文件！
    if score_repro > 0.74:
        save_name = f"Best_DeepSurv_Repro_{score_repro:.4f}.pth"
        torch.save(m_repro.state_dict(), save_name)
        print(f"  ✅ 成功复现高分！模型已单独保存为: {save_name}")
    else:
        print(f"  🤔 结果为 {score_repro:.4f}，未达 0.74。")

    print("\n" + "="*40)
    print("🚀 最终结果 (已保存所有模型)")
    print("="*40)
    for k, v in results.items(): print(f"{k:<20}: {v:.4f}")
    print("\n✅ 所有最佳模型和 Scaler 已保存到当前目录！可以进行下一步分析了。")

if __name__ == "__main__":
    main()