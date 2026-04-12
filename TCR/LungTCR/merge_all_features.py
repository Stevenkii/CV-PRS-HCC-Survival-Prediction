import pandas as pd
import glob
import os

# ================= 配置区域 =================
# 1. 多样性文件 (Step 1): 使用 Resampled 版本
DIV_FILE = "output_diversity/diversity.strict.resampled.txt"

# 2. 特征文件夹 (Step 2)
FEAT_DIR = "output_features"

# 3. 输出文件名
OUTPUT_FILE = "test.csv"
# ===========================================

print("🚀 开始执行全量特征合并...")

# --- 第一步：读取多样性特征 (基准表) ---
if os.path.exists(DIV_FILE):
    # 读取 txt, 也就是 Step 1 的结果
    base_df = pd.read_csv(DIV_FILE, sep='\t')
    
    # 标准化样本列名：VDJtools 输出可能是 'sample_id' 或 'sample'
    # 我们统一改成 'sample_id' 以便后续合并
    if 'sample_id' not in base_df.columns:
        for col in ['sample', 'Sample', 'SampleID']:
            if col in base_df.columns:
                base_df.rename(columns={col: 'sample_id'}, inplace=True)
                break
    
    print(f"✅ 基准表加载成功: {os.path.basename(DIV_FILE)}")
    print(f"   样本数: {base_df.shape[0]}, 初始特征数: {base_df.shape[1]}")
else:
    print(f"❌ 错误: 找不到文件 {DIV_FILE}")
    exit()

# --- 第二步：循环读取并合并所有特征 CSV ---
# 找到 output_features 下所有的 .csv 文件
feat_files = glob.glob(os.path.join(FEAT_DIR, "*.csv"))
feat_files.sort() # 排序，保证顺序一致

final_df = base_df

for f in feat_files:
    fname = os.path.basename(f)
    try:
        # 读取 CSV
        df = pd.read_csv(f)
        
        # 寻找用于合并的 Key (样本ID列)
        merge_key = None
        for col in ['sample_id', 'SampleID', 'sample', 'Sample']:
            if col in df.columns:
                merge_key = col
                break
        
        if merge_key:
            # 如果列名不叫 sample_id，统一改名
            if merge_key != 'sample_id':
                df.rename(columns={merge_key: 'sample_id'}, inplace=True)
            
            # 执行合并 (Left Join: 保证不丢失基准表的样本)
            # suffix 参数防止列名冲突 (比如两个表都有 total_reads)
            final_df = pd.merge(final_df, df, on='sample_id', how='left', suffixes=('', f'_{fname}'))
            print(f"🔗 合并成功: {fname} (新增特征列: {df.shape[1]-1})")
        else:
            print(f"⚠️ 跳过文件 {fname}: 未找到 sample_id 列")
            
    except Exception as e:
        print(f"❌ 读取失败 {fname}: {e}")

# --- 第三步：数据清洗与保存 ---
# 1. 填充 NA 为 0 (机器学习模型通常不接受空值)
# 之前的讨论确认过，这里的 NA 通常意味着“没测到”，在生物学上等同于 0
final_df.fillna(0, inplace=True)

# 2. 保存结果
final_df.to_csv(OUTPUT_FILE, index=False)

print("-" * 30)
print(f"🎉 合并全部完成！")
print(f"📊 最终矩阵维度: {final_df.shape} (行=样本数, 列=特征数)")
print(f"💾 文件保存在: {os.path.abspath(OUTPUT_FILE)}")