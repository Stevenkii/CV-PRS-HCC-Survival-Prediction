import os
import glob
import h5py
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, ParameterGrid
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

# ================= 1. 配置区域 =================

TRAIN_CSV = "ready_to_train_train.csv"
VAL_CSV   = "ready_to_train_val.csv"
TEST_CSV  = "ready_to_train_test.csv"

# 图像特征文件夹路径
TRAIN_FEAT_DIR = "/haplox/users/xiax/project/HCC_imagene/TCGA_data/train_trident_processed/20x_512px_0px_overlap/slide_features_titan"
VAL_FEAT_DIR   = "/haplox/users/xiax/project/HCC_imagene/TCGA_data/val_trident_processed/20x_512px_0px_overlap/slide_features_titan"
TEST_FEAT_DIR  = "/haplox/users/xiax/project/HCC_imagene/TCGA_data/test_trident_processed/20x_512px_0px_overlap/slide_features_titan"

ID_COL = "Sample_ID"
TIME_COL = "survival_time"
EVENT_COL = "survival_status"

# ================= 2. 数据加载函数 (严格模式) =================

def get_combined_file_mapping(dir_list):
    mapping = {}
    print(f"\n[Step 1] 建立全局文件索引...")
    for folder in dir_list:
        if not os.path.exists(folder): continue
        h5_files = glob.glob(os.path.join(folder, "*.h5"))
        for path in h5_files:
            fname = os.path.basename(path)
            short_id = fname[:12]
            mapping[short_id] = path
    print(f"✅ 索引完成，共索引 {len(mapping)} 个文件。\n")
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
                time = float(row[TIME_COL])
                event = int(row[EVENT_COL])
                valid_data.append((np.hstack((clin, feat)), time, event))
            except:
                pass
    
    if not valid_data: return None, None, None
    X = np.array([x[0] for x in valid_data], dtype=np.float32)
    T = np.array([x[1] for x in valid_data], dtype=float)
    E = np.array([x[2] for x in valid_data], dtype=int)
    return X, T, E

# ================= 3. 核心流程：广域 5折交叉验证 =================

def main():
    all_dirs = [TRAIN_FEAT_DIR, VAL_FEAT_DIR, TEST_FEAT_DIR]
    file_mapping = get_combined_file_mapping(all_dirs)
    
    print("[Step 2] 加载数据...")
    X_train, T_train, E_train = load_data_strict(TRAIN_CSV, file_mapping)
    X_val,   T_val,   E_val   = load_data_strict(VAL_CSV, file_mapping)
    X_test,  T_test,  E_test  = load_data_strict(TEST_CSV, file_mapping)
    
    # === 关键步骤：合并 训练(195) + 验证(65) = 开发集(260) ===
    X_dev = np.vstack((X_train, X_val))
    T_dev = np.concatenate((T_train, T_val))
    E_dev = np.concatenate((E_train, E_val))
    
    print(f"  开发集样本数: {len(X_dev)} (用于5折交叉验证)")
    print(f"  测试集样本数: {len(X_test)} (用于最终独立测试)")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # ==========================================
    # 模型 A: Random Survival Forest (RSF)
    # ==========================================
    print("\n" + "="*60)
    print("模型 A: RSF - 深度参数搜索")
    print("="*60)
    
    rsf_param_grid = {
        'n_estimators': [1000],            # 足够多
        'min_samples_leaf': [5, 10, 15],   # 覆盖过拟合到欠拟合
        'max_features': ['sqrt', 'log2'],  # 高维数据关键
        'max_depth': [None, 10]            # 限制深度
    }
    
    rsf_grid = list(ParameterGrid(rsf_param_grid))
    print(f"共生成 {len(rsf_grid)} 种参数组合...\n")
    
    best_rsf_score = -1
    best_rsf_params = None
    
    for i, params in enumerate(rsf_grid):
        cv_scores = []
        for train_idx, val_idx in kf.split(X_dev):
            X_tr, X_val_fold = X_dev[train_idx], X_dev[val_idx]
            
            y_tr = np.zeros(len(train_idx), dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
            y_tr['Status'] = E_dev[train_idx].astype(bool)
            y_tr['Survival_in_days'] = T_dev[train_idx]
            
            y_val_fold = np.zeros(len(val_idx), dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
            y_val_fold['Status'] = E_dev[val_idx].astype(bool)
            y_val_fold['Survival_in_days'] = T_dev[val_idx]
            
            model = RandomSurvivalForest(n_jobs=-1, random_state=42, **params)
            model.fit(X_tr, y_tr)
            
            try:
                cv_scores.append(model.score(X_val_fold, y_val_fold))
            except:
                cv_scores.append(0.5)
        
        avg_score = np.mean(cv_scores)
        if avg_score > best_rsf_score:
            best_rsf_score = avg_score
            best_rsf_params = params
            print(f"[{i+1}/{len(rsf_grid)}] 新最佳! {params} -> CV: {avg_score:.4f}")
        elif (i+1) % 5 == 0:
            print(f"[{i+1}/{len(rsf_grid)}] 当前: {params} -> {avg_score:.4f}")

    print(f"\n🏆 RSF 最佳参数: {best_rsf_params}")
    
    # RSF 全量重训
    print("🚀 RSF 全量重训中...")
    y_dev = np.zeros(len(T_dev), dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    y_dev['Status'] = E_dev.astype(bool)
    y_dev['Survival_in_days'] = T_dev
    final_rsf = RandomSurvivalForest(n_jobs=-1, random_state=42, **best_rsf_params)
    final_rsf.fit(X_dev, y_dev)
    
    y_test = np.zeros(len(T_test), dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    y_test['Status'] = E_test.astype(bool)
    y_test['Survival_in_days'] = T_test
    print(f"🚀 RSF 测试集最终 C-Index: {final_rsf.score(X_test, y_test):.4f}")


    # ==========================================
    # 模型 B: XGBoost Survival (增强版)
    # ==========================================
    print("\n" + "="*60)
    print("模型 B: XGBoost - 增强参数搜索 (含正则化)")
    print("="*60)
    
    def get_label(t, e): return np.where(e == 1, t, -t)
    
    # 【新增正则化参数】
    # gamma: 树分裂的最小损失下降值，值越大越保守
    # reg_alpha (L1): 权重稀疏化，适合高维特征筛选
    xgb_param_grid = {
        'eta': [0.01, 0.03],           # 学习率
        'max_depth': [2, 3],           # 树深
        'subsample': [0.7],            # 保持0.7即可
        'colsample_bytree': [0.5, 0.7, 1.0],# 特征采样
        'gamma': [0, 0.2],             # <--- 新增
        'reg_alpha': [0, 0.2, 0.5, 1]            # <--- 新增 (L1正则)
    }
    
    xgb_grid = list(ParameterGrid(xgb_param_grid))
    print(f"共生成 {len(xgb_grid)} 种参数组合...\n")
    
    best_xgb_score = -1
    best_xgb_params = None
    
    for i, params in enumerate(xgb_grid):
        cv_scores = []
        for train_idx, val_idx in kf.split(X_dev):
            dtr = xgb.DMatrix(X_dev[train_idx], label=get_label(T_dev[train_idx], E_dev[train_idx]))
            dval = xgb.DMatrix(X_dev[val_idx], label=get_label(T_dev[val_idx], E_dev[val_idx]))
            
            bp = {
                'objective': 'survival:cox',
                'eval_metric': 'cox-nloglik',
                'tree_method': 'hist',
                'seed': 42,
                **params
            }
            # 适当减少 boost rounds 以加快搜索，最终训练再加满
            model = xgb.train(bp, dtr, num_boost_round=1000, 
                              evals=[(dval, 'v')], early_stopping_rounds=20, verbose_eval=False)
            
            preds = model.predict(dval)
            try:
                res = concordance_index_censored(E_dev[val_idx].astype(bool), T_dev[val_idx], preds)
                cv_scores.append(res[0])
            except:
                cv_scores.append(0.5)
                
        avg_score = np.mean(cv_scores)
        if avg_score > best_xgb_score:
            best_xgb_score = avg_score
            best_xgb_params = params
            print(f"[{i+1}/{len(xgb_grid)}] 新最佳! {params} -> CV: {avg_score:.4f}")
        elif (i+1) % 10 == 0:
            print(f"[{i+1}/{len(xgb_grid)}] 当前: {params} -> {avg_score:.4f}")

    print(f"\n🏆 XGB 最佳参数: {best_xgb_params}")
    
    # XGB 全量重训
    print("🚀 XGB 全量重训中...")
    final_bp = {
        'objective': 'survival:cox', 'eval_metric': 'cox-nloglik',
        'tree_method': 'hist', 'seed': 42, **best_xgb_params
    }
    d_dev_full = xgb.DMatrix(X_dev, label=get_label(T_dev, E_dev))
    d_test_full = xgb.DMatrix(X_test, label=get_label(T_test, E_test))
    
    # 增加轮数，确保收敛
    final_xgb = xgb.train(final_bp, d_dev_full, num_boost_round=1500, verbose_eval=False)
    test_preds = final_xgb.predict(d_test_full)
    print(f"🚀 XGB 测试集最终 C-Index: {concordance_index_censored(E_test.astype(bool), T_test, test_preds)[0]:.4f}")

if __name__ == "__main__":
    main()