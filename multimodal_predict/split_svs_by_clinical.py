import os
import shutil
import pandas as pd
from pathlib import Path
import argparse


def organize_svs_files(svs_dir, output_dir, train_tsv, val_tsv, test_tsv):
    """
    根据 clinical tsv 文件中的 patient ID (case_submitter_id) 将 SVS 文件
    整理到 train, val, test 子文件夹中。
    """
    
    # 1. 读取三个 TSV 文件，提取 case_submitter_id
    print("正在读取 Clinical TSV 文件...")
    
    datasets = {
        'train': train_tsv,
        'val': val_tsv,
        'test': test_tsv
    }
    
    # 构建 映射表: patient_id -> dataset_type ('train', 'val', 'test')
    patient_to_split = {}
    
    for split_name, tsv_path in datasets.items():
        if not os.path.exists(tsv_path):
            print(f"[警告] 文件不存在: {tsv_path}")
            continue
            
        try:
            df = pd.read_csv(tsv_path, sep='\t')
            # 兼容可能的列名 (case_submitter_id 或 submitter_id)
            col_name = 'case_submitter_id' if 'case_submitter_id' in df.columns else 'submitter_id'
            
            if col_name in df.columns:
                pids = df[col_name].dropna().astype(str).unique()
                print(f"  > {split_name}: 包含 {len(pids)} 个病人")
                for pid in pids:
                    patient_to_split[pid] = split_name
            else:
                print(f"[错误] {tsv_path} 中未找到病人ID列")
        except Exception as e:
            print(f"[错误] 读取 {tsv_path} 失败: {e}")

    print(f"总共加载了 {len(patient_to_split)} 个病人的分组信息。")

    # 2. 准备输出目录
    output_path = Path(output_dir)
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    # 3. 遍历 SVS 目录中的文件
    print(f"\n开始扫描并移动文件: {svs_dir} ...")
    svs_path = Path(svs_dir)
    
    processed_count = 0
    moved_count = 0
    unknown_count = 0
    
    # 遍历所有文件 (包括子目录)
    # 你的文件名格式示例: TCGA-2Y-A9GT-01Z-00-DX1.30666775-3556-4DFE-A5EC-8CCF8EEB1803.svs
    # 病人ID通常是前12位: TCGA-2Y-A9GT
    
    for file_path in svs_path.rglob('*.svs'):
        processed_count += 1
        filename = file_path.name
        
        # 提取病人ID (前12个字符)
        # TCGA-xx-xxxx
        patient_id = filename[:12]
        
        if patient_id in patient_to_split:
            split = patient_to_split[patient_id]
            dest_folder = output_path / split
            dest_file = dest_folder / filename
            
            # 执行移动 (或拷贝，这里用 move 移动，如果想保留原文件可用 copy2)
            # 建议先用 copy 测试，确认无误后再改 move
            try:
                if not dest_file.exists():
                    shutil.copy2(file_path, dest_file) # 这里使用 copy2 (复制)
                    # shutil.move(file_path, dest_file) # 如果想移动，取消注释这行
                    moved_count += 1
                    if moved_count % 10 == 0:
                        print(f"  已处理 {moved_count} 个文件...", end='\r')
                else:
                    # print(f"[跳过] 目标已存在: {filename}")
                    pass
            except Exception as e:
                print(f"[错误] 移动失败 {filename}: {e}")
        else:
            unknown_count += 1
            # print(f"[未知] 找不到对应的分组: {patient_id} ({filename})")

    print("\n" + "="*30)
    print("处理完成")
    print(f"扫描 SVS 文件数: {processed_count}")
    print(f"成功分发文件数: {moved_count}")
    print(f"未匹配到分组数: {unknown_count}")
    
    if unknown_count > 0:
        print("\n提示: 未匹配的文件可能是因为 TSV 文件中没有包含该病人的记录，或者文件名格式不匹配。")

if __name__ == "__main__":
    # 配置你的路径
    # SVS_SOURCE_DIR: 刚才你下载好的存放所有 SVS 的文件夹
    # OUTPUT_DIR: 你希望生成 train/val/test 的根目录
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--svs_dir", required=True, help="存放 SVS 文件的源目录")
    parser.add_argument("--output_dir", required=True, help="输出目录 (将自动创建 train/val/test)")
    parser.add_argument("--train_tsv", default="clinical_train.tsv", help="训练集 TSV 路径")
    parser.add_argument("--val_tsv", default="clinical_val.tsv", help="验证集 TSV 路径")
    parser.add_argument("--test_tsv", default="clinical_test.tsv", help="测试集 TSV 路径")
    
    args = parser.parse_args()
    
    organize_svs_files(args.svs_dir, args.output_dir, args.train_tsv, args.val_tsv, args.test_tsv)
