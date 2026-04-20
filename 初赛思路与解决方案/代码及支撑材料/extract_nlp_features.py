import json
import os
import multiprocessing
import time
from tqdm import tqdm

def process_single_line_static(line):
    """
    核心特征提取函数（并行执行单元）。
    严格遵循原始过滤逻辑。
    """
    try:
        data = json.loads(line.strip())
        raw_text = data.get("text_sequence", "")
        raw_lines = raw_text.split('\n')
        semantic_events = []
        last_timestamp = ""
        
        for r_line in raw_lines:
            parts = r_line.split('|')
            if len(parts) < 2: 
                continue
            timestamp, event_type = parts[0], parts[1]
            
            # 原始语义过滤逻辑
            if event_type in ["游戏开始", "动作", "伤害", "技能", "可搜索的散点物资"]:
                semantic_events.append(r_line)
            elif event_type == "玩家基础信息":
                if timestamp.endswith(".00") and timestamp != last_timestamp:
                    semantic_events.append(r_line)
                    last_timestamp = timestamp
        
        compressed_text = "\n".join(semantic_events)
        out_data = {
            "file_name": data.get("file_name", "unknown"),
            "nlp_text": compressed_text,
            "main_label": data.get("main_label"),
            "sub_label": data.get("sub_label")
        }
        return json.dumps(out_data, ensure_ascii=False) + "\n"
    except Exception:
        return None

def line_generator(file_path):
    """
    流式读取生成器，防止大文件撑爆内存。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield line

def run_extraction():
    input_file = "structured_dataset.jsonl"
    output_file = "nlp_ready_dataset.jsonl"
    num_workers = 16  # 建议保留 2-4 个核心给系统，设置 16-18 即可
    
    if not os.path.exists(input_file):
        print(f"ERROR: 找不到输入文件 {input_file}")
        return

    # 获取文件总行数（用于进度条显示，若文件极大可跳过此步）
    print("INFO: 正在预估文件规模...")
    # 使用快速统计方法
    total_lines = sum(1 for _ in open(input_file, 'r'))
    print(f"INFO: 任务启动。待处理记录: {total_lines} 条 | 并行核心数: {num_workers}")

    start_time = time.time()
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        with multiprocessing.Pool(processes=num_workers) as pool:
            # 使用 imap 以保持写入顺序，并配合 tqdm 显示进度条
            # chunksize=500 可减少进程间通讯频率，提升吞吐量
            iterator = pool.imap(process_single_line_static, line_generator(input_file), chunksize=500)
            
            for result_str in tqdm(iterator, total=total_lines, desc="Feature Extracting", unit="record"):
                if result_str:
                    f_out.write(result_str)

    duration = time.time() - start_time
    print(f"INFO: 特征提取完成。总耗时: {duration:.2f}s | 平均速度: {total_lines/duration:.2f} records/s")

if __name__ == "__main__":
    run_extraction()