import os
import glob
import h5py
import numpy as np
import pandas as pd
import xgboost as xgb
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

# ================= 1. 配置区域 (请确认路径无误) =================

# CSV 文件路径
TRAIN_CSV = "ready_to_train_train.csv"
VAL_CSV   = "ready_to_train_val.csv"
TEST_CSV  = "ready_to_train_test.csv"

# 图像特征文件夹路径 (分别对应三个集合)
# 程序会自动扫描这些文件夹里的所有 .h5 文件
TRAIN_FEAT_DIR = "/haplox/users/xiax/project/HCC_imagene/TCGA_data/train_trident_processed/20x_512px_0px_overlap/slide_features_titan"
VAL_FEAT_DIR   = "/haplox/users/xiax/project/HCC_imagene/TCGA_data/val_trident_processed/20x_512px_0px_overlap/slide_features_titan"
TEST_FEAT_DIR  = "/haplox/users/xiax/project/HCC_imagene/TCGA_data/test_trident_processed/20x_512px_0px_overlap/slide_features_titan"

# CSV 中的关键列名
ID_COL = "Sample_ID"
TIME_COL = "survival_time"
EVENT_COL = "survival_status"

# ================= 2. 工具函数：建立文件索引 =================

def get_combined_file_mapping(dir_list):
    """
    扫描多个文件夹，建立 {Sample_ID前12位: .h5文件绝对路径} 的字典。
    这样无论样本在哪里，只要ID对得上就能找到。
    """
    mapping = {}
    print(f"\n[Step 1] 正在建立文件索引...")
    
    for folder in dir_list:
        if not os.path.exists(folder):
            print(f"⚠️  警告: 文件夹不存在 -> {folder}")
            continue
        
        # 查找所有 .h5 文件
        h5_files = glob.glob(os.path.join(folder, "*.h5"))
        for path in h5_files:
            fname = os.path.basename(path)
            # 截取 TCGA ID 的前12位 (例如 TCGA-2Y-A9GS)
            short_id = fname[:12]
            mapping[short_id] = path
            
    print(f"✅ 索引建立完成，共索引 {len(mapping)} 个唯一样本。\n")
    return mapping

# ================= 3. 核心函数：加载数据 (严格匹配模式) =================

def load_data(csv_path, file_mapping):
    """
    读取 CSV 并拼接图像特征。
    【严格模式】：如果找不到对应的图像特征，直接丢弃该样本，不进行零填充。
    """
    print(f"正在处理: {csv_path} ...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # 临床特征列 (排除 ID, Time, Event)
    exclude_cols = [ID_COL, TIME_COL, EVENT_COL]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # 临时列表，用于存储有效的样本
    valid_clinical = []
    valid_imgs = []
    valid_times = []
    valid_events = []
    
    drop_count = 0
    
    for idx, row in df.iterrows():
        sample_id = row[ID_COL]
        # 确保ID转为字符串并取前12位
        short_id = str(sample_id)[:12]
        
        # 在索引中查找 .h5 路径
        h5_path = file_mapping.get(short_id)
        
        # === 核心逻辑：只有当文件存在且能读取时，才保留样本 ===
        if h5_path and os.path.exists(h5_path):
            try:
                feat = None
                with h5py.File(h5_path, 'r') as f:
                    # 尝试读取 'features' 键，如果不存在则读取第一个键
                    if 'features' in f:
                        feat = f['features'][:]
                    else:
                        key = list(f.keys())[0]
                        feat = f[key][:]
                
                # 只有成功读取到这里，才把数据加入列表
                valid_imgs.append(feat)
                # 临床特征转为 float32
                valid_clinical.append(row[feature_cols].values.astype(np.float32))
                valid_times.append(float(row[TIME_COL]))
                valid_events.append(int(row[EVENT_COL]))
                
            except Exception as e:
                print(f"  读取损坏 {short_id}: {e}")
                drop_count += 1
        else:
            # 文件不存在，丢弃该样本
            drop_count += 1
            
    if drop_count > 0:
        print(f"  ⚠️  已丢弃 {drop_count} 个样本 (原因: 缺少对应的图像特征文件)")
        
    # 检查是否所有样本都被丢弃了
    if len(valid_imgs) == 0:
        raise ValueError(f"错误：{csv_path} 中没有一个样本匹配到了图像特征！请检查路径配置。")

    # 转换为 Numpy 数组
    X_clin = np.array(valid_clinical, dtype=np.float32)
    X_img = np.array(valid_imgs, dtype=np.float32)
    
    # 特征拼接 [N, 临床特征数 + 768]
    X_combined = np.hstack((X_clin, X_img))
    
    y_time = np.array(valid_times)
    y_event = np.array(valid_events)
    
    # 制作 RSF 专用结构化标签
    # 格式: [(Status, Time), ...]
    y_structured = np.zeros(len(y_time), dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    y_structured['Status'] = y_event.astype(bool)
    y_structured['Survival_in_days'] = y_time
    
    print(f"  -> 有效样本数: {len(X_combined)}")
    return X_combined, y_structured, (y_time, y_event)

# ================= 4. 主流程 =================

def main():
    # --- Step 1: 建立全局索引 ---
    all_dirs = [TRAIN_FEAT_DIR, VAL_FEAT_DIR, TEST_FEAT_DIR]
    file_mapping = get_combined_file_mapping(all_dirs)
    
    # --- Step 2: 加载数据 ---
    # 这里加载的数据已经是剔除了坏样本的“纯净版”
    X_train, y_train_struc, (yt_train, ye_train) = load_data(TRAIN_CSV, file_mapping)
    X_val,   y_val_struc,   (yt_val, ye_val)     = load_data(VAL_CSV, file_mapping)
    X_test,  y_test_struc,  (yt_test, ye_test)   = load_data(TEST_CSV, file_mapping)
    
    print(f"\n特征总维度: {X_train.shape[1]} (其中768维是图像特征)")
    
    # ==========================================
    # 模型 A: Random Survival Forest (RSF)
    # ==========================================
    print("\n" + "="*50)
    print("模型 A: Random Survival Forest (RSF) - 参数搜索")
    print("="*50)
    
    # 【参数策略】
    # 针对 TCGA 这种高维(798维) + 小样本(几百例)的数据：
    # 1. n_estimators 要大 (500-1000)，保证森林足够茂盛，结果稳定。
    # 2. min_samples_leaf 要大 (10-20)，强制每片叶子至少有10-20个病人，防止死记硬背。
    rsf_grid = [
        {'n_estimators': 500,  'min_samples_leaf': 10},  # 方案1: 均衡
        {'n_estimators': 1000, 'min_samples_leaf': 10},  # 方案2: 更加稳定
        {'n_estimators': 500,  'min_samples_leaf': 20},  # 方案3: 强正则化 (防过拟合)
        {'n_estimators': 1000, 'min_samples_leaf': 20},  # 方案4: 大且稳
    ]
    
    best_rsf_score = -1
    best_rsf_model = None
    best_rsf_config = {}
    
    for params in rsf_grid:
        print(f"尝试参数: {params} ...")
        # n_jobs=-1 代表使用所有 CPU 核心加速
        model = RandomSurvivalForest(n_jobs=-1, random_state=42, **params)
        model.fit(X_train, y_train_struc)
        
        # 在验证集上评估
        score = model.score(X_val, y_val_struc)
        print(f"  -> 验证集 C-Index: {score:.4f}")
        
        if score > best_rsf_score:
            best_rsf_score = score
            best_rsf_model = model
            best_rsf_config = params
            
    print(f"\n🏆 RSF 最佳参数: {best_rsf_config}")
    print(f"🏆 RSF 验证集最佳得分: {best_rsf_score:.4f}")
    
    # 最终测试
    final_score = best_rsf_model.score(X_test, y_test_struc)
    print(f"🚀 RSF 测试集最终 C-Index: {final_score:.4f}")


    # ==========================================
    # 模型 B: XGBoost Survival (Cox Objective)
    # ==========================================
    print("\n" + "="*50)
    print("模型 B: XGBoost Survival (Cox) - 参数搜索")
    print("="*50)
    
    # 【XGBoost 标签制作】
    # Cox Loss 需要知道哪些是 Event(死), 哪些是 Censored(活)
    # 技巧: 正数代表死于该时间，负数代表在该时间删失
    # 比如: 死于100天 -> 100; 活过100天(删失) -> -100
    def make_dmatrix(X, t, e):
        label = np.where(e == 1, t, -1.0 * t) 
        return xgb.DMatrix(X, label=label)
    
    dtrain = make_dmatrix(X_train, yt_train, ye_train)
    dval   = make_dmatrix(X_val, yt_val, ye_val)
    dtest  = make_dmatrix(X_test, yt_test, ye_test)
    
    # 【参数策略】
    # max_depth: 2-4。医学数据噪声大，树太深会过拟合。浅树(2-3层)往往效果最好。
    # eta: 0.01-0.05。学习率要慢，配合大轮数(num_boost_round)，能找到更细致的最优解。
    # subsample: 0.7-0.8。每次只用70%-80%的样本建树，增加随机性，防过拟合。
    xgb_grid = [
        {'eta': 0.01, 'max_depth': 2, 'subsample': 0.7}, # 极简模式 (防过拟合最强)
        {'eta': 0.01, 'max_depth': 3, 'subsample': 0.7}, # 标准模式
        {'eta': 0.05, 'max_depth': 3, 'subsample': 0.8}, # 加速模式
        {'eta': 0.05, 'max_depth': 4, 'subsample': 0.8}, # 复杂模式
    ]
    
    best_xgb_score = -1
    best_xgb_model = None
    best_xgb_config = {}
    
    for params in xgb_grid:
        print(f"尝试参数: {params} ...")
        train_params = {
            'objective': 'survival:cox',    # 目标函数: Cox回归
            'eval_metric': 'cox-nloglik',   # 评估指标
            'tree_method': 'hist',          # 加速直方图算法
            'seed': 42,
            **params
        }
        
        # 训练
        model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=2000,         # 最大轮数 (配合early_stopping使用)
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,     # 如果验证集50轮没提升就停
            verbose_eval=False            # 不刷屏
        )
        
        # 验证
        val_preds = model.predict(dval)
        # XGB 输出的是 log hazard (风险评分)，越高风险越大
        res = concordance_index_censored(y_val_struc['Status'], y_val_struc['Survival_in_days'], val_preds)
        val_c = res[0]
        print(f"  -> 验证集 C-Index: {val_c:.4f}")
        
        if val_c > best_xgb_score:
            best_xgb_score = val_c
            best_xgb_model = model
            best_xgb_config = params
            
    print(f"\n🏆 XGB 最佳参数: {best_xgb_config}")
    print(f"🏆 XGB 验证集最佳得分: {best_xgb_score:.4f}")
    
    # 最终测试
    test_preds = best_xgb_model.predict(dtest)
    final_res = concordance_index_censored(y_test_struc['Status'], y_test_struc['Survival_in_days'], test_preds)
    print(f"🚀 XGB 测试集最终 C-Index: {final_res[0]:.4f}")

if __name__ == "__main__":
    main()