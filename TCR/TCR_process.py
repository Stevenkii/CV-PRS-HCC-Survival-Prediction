import pandas as pd
import glob
import os

def process_batch2_final(file_path):
    filename = os.path.basename(file_path)
    print(f"--------------------------------")
    print(f"📄 正在处理: {filename}")
    
    try:
        df = pd.read_csv(file_path, sep='\t')
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 1. 验证是否为第二批数据
    if 'duplicate_count' not in df.columns:
        print(f"⏭️ 跳过: 不是第二批数据格式")
        return

    new_df = pd.DataFrame()

    # === 1. 核心数值转换 ===
    # Count: 直接映射
    new_df['count'] = df['duplicate_count']

    # Freq: 自动计算 (Count / Total)
    total_reads = new_df['count'].sum()
    new_df['freq'] = new_df['count'] / total_reads
    print(f"   📊 总Count: {total_reads} (Freq已计算)")

    # === 2. 序列提取 ===
    # AIRR格式: junction -> cdr3nt, junction_aa -> cdr3aa
    new_df['cdr3nt'] = df['junction'] if 'junction' in df.columns else '.'
    new_df['cdr3aa'] = df['junction_aa'] if 'junction_aa' in df.columns else '.'

    # === 3. 基因名清洗 (逻辑: 取第一个，去星号) ===
    # 虽然你这批数据没有逗号，但保留 split(',')[0] 是为了代码的通用性和健壮性
    def clean_gene(val):
        if pd.isna(val) or val == '.': return '.'
        return str(val).split(',')[0].split('*')[0].split('(')[0]

    gene_map = {'v': 'v_call', 'd': 'd_call', 'j': 'j_call'}
    for out_col, in_col in gene_map.items():
        new_df[out_col] = df[in_col].apply(clean_gene) if in_col in df.columns else '.'

    # === 4. 结构坐标 (VEnd, JStart 填真实值, 其他填 -1) ===
    # 这是最完美的处理方式，既保留了信息，又符合格式要求
    
    # VEnd (v_junction_end)
    if 'v_junction_end' in df.columns:
        new_df['VEnd'] = df['v_junction_end'].fillna(-1).astype(int)
    else:
        new_df['VEnd'] = -1

    # JStart (j_junction_start)
    if 'j_junction_start' in df.columns:
        new_df['JStart'] = df['j_junction_start'].fillna(-1).astype(int)
    else:
        new_df['JStart'] = -1

    # DStart, DEnd (缺失，填 -1)
    new_df['DStart'] = -1
    new_df['DEnd'] = -1

    # === 5. 输出保存 ===
    # 按照 VDJtools 官方列序排列
    cols_order = ['count', 'freq', 'cdr3nt', 'cdr3aa', 'v', 'd', 'j', 
                  'VEnd', 'DStart', 'DEnd', 'JStart']
    
    # 防呆检查：确保所有列都存在
    for c in cols_order:
        if c not in new_df.columns:
            new_df[c] = -1 if c in ['VEnd', 'DStart', 'DEnd', 'JStart'] else '.'

    new_df = new_df[cols_order]
    
    output_name = "VDJtools_" + filename.replace('.clonotypes.TRB.txt', '.txt')
    new_df.to_csv(output_name, sep='\t', index=False)
    print(f"✅ 转换完成: {output_name}")

# --- 主程序 ---
files = glob.glob("*_T.clonotypes.TRB.txt")
print(f"🔍 开始批量处理 {len(files)} 个文件...")

for f in files:
    # 避免重复处理生成的文件
    if f.startswith("VDJtools_"):
        continue
    process_batch2_final(f)