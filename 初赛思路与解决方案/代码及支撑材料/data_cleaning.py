import os
import glob
import json

def clean_and_build_dataset():
    """
    遍历训练数据目录，读取所有的txt文件，清洗空行，并打上对应的大类和小类标签，
    最终生成 structured_dataset.jsonl 以供大语言模型(LLM)或序列模型训练使用。
    """
    # 修复核心Bug: 将基准路径改为当前目录
    base_dir = "."
    output_file = "structured_dataset.jsonl"
    
    # 标签映射字典
    main_label_map = {
        'classified_samples_0': 0, # 交战类
        'classified_samples_1': 1  # 避战类
    }
    
    sub_label_map = {
        'Action': 0,
        'BeingResuce': 1,  # 目录名拼写保持原样匹配
        'Fire': 2,
        'Grenade': 3,
        'Looting': 4,
        'SkillStart': 5
    }
    
    total_processed = 0
    error_count = 0
    
    print(f"INFO: 开始数据清洗，输出文件将保存至 {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for main_folder, main_label in main_label_map.items():
            for sub_folder, sub_label in sub_label_map.items():
                # 构造路径匹配模式
                pattern = os.path.join(base_dir, main_folder, sub_folder, "*.txt")
                files = glob.glob(pattern)
                
                for file_path in files:
                    try:
                        # 读取文本
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            
                        # 数据清洗：去除空行和前后空白字符
                        cleaned_lines = [line.strip() for line in lines if line.strip()]
                        
                        # 重组为单一文本序列
                        text_sequence = "\n".join(cleaned_lines)
                        
                        # 构造数据集条目
                        data_item = {
                            "file_name": os.path.basename(file_path),
                            "text_sequence": text_sequence,
                            "main_label": main_label,
                            "sub_label": sub_label
                        }
                        
                        # 写入 JSONL (一行一个 JSON 对象)
                        out_f.write(json.dumps(data_item, ensure_ascii=False) + "\n")
                        total_processed += 1
                        
                        if total_processed % 1000 == 0:
                            print(f"INFO: 已处理 {total_processed} 个文件...")
                            
                    except Exception as e:
                        error_count += 1
                        print(f"ERROR: 读取文件失败: {file_path}, 错误: {e}")
                        
    print(f"INFO: 数据清洗完成。成功处理文件总数: {total_processed}, 失败数: {error_count}")

if __name__ == "__main__":
    clean_and_build_dataset()