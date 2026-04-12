import pandas as pd
import glob
import os

# ================= 配置 =================
DATA_DIR = "/public/home/lifex/xiax/HCC_TCGA_data/TCR/3people/LungTCR/1stbatch"
OUTPUT_FILE = "metadata.tsv"
# =======================================

print(f"📂 正在扫描目录: {DATA_DIR}")

# 1. 获取所有相关文件 (兼容你的两种命名格式)
files_type_a = glob.glob(os.path.join(DATA_DIR, "VDJtools_*.txt"))
files_type_b = glob.glob(os.path.join(DATA_DIR, "vdjtools_result_*.txt"))

# 合并并去重
all_files = sorted(list(set(files_type_a + files_type_b)))

if not all_files:
    print("❌ 未找到任何文件！")
    exit()

print(f"🔍 找到 {len(all_files)} 个文件")

metadata = []

for file_path in all_files:
    filename = os.path.basename(file_path)
    sample_id = ""

    # === 智能提取样本名 ===
    
    # 格式 A: VDJtools_chenwuxuan_T.txt
    if filename.startswith("VDJtools_"):
        temp = filename.replace("VDJtools_", "") 
        sample_id = temp.split('_T')[0]
        
    # 格式 B: vdjtools_result_litianyou_T.litianyou_T...
    elif filename.startswith("vdjtools_result_"):
        temp = filename.replace("vdjtools_result_", "")
        sample_id = temp.split('_T')[0]
    
    else:
        sample_id = filename.split('.')[0]

    # === 构建元数据 (严格只保留两列) ===
    metadata.append({
        'file_name': os.path.abspath(file_path), # 绝对路径
        'sample_id': sample_id                     # 样本ID
    })
    
    print(f"  {sample_id} <--- {filename}")

# 2. 生成 DataFrame
df = pd.DataFrame(metadata)

# 3. 严格只保留这两列，且不带表头索引
# 注意：LungTCR 某些脚本可能需要表头，通常建议保留 header
df = df[['file_name', 'sample_id']]

# 4. 保存 (Tab分隔)
df.to_csv(OUTPUT_FILE, sep='\t', index=False)

print("-" * 30)
print(f"✅ metadata.tsv 生成完毕！(无Label列)")
print(f"📄 路径: {os.path.abspath(OUTPUT_FILE)}")