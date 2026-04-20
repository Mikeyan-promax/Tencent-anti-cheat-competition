import os
import glob
import json
import logging
import random
import pickle
from collections import defaultdict
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_denoised_inputs(lines):
    semantic_events =[]
    last_timestamp = ""
    for line in lines:
        line = line.strip()
        if not line: continue
        parts = line.split('|')
        if len(parts) < 2: continue
        try:
            ts = float(parts[0])
            if ts > 20.0: break
        except ValueError:
            continue
            
        event_type = parts[1]
        if event_type in["游戏开始", "动作", "伤害", "技能", "可搜索的散点物资"]:
            semantic_events.append(line)
        elif event_type == "玩家基础信息":
            if parts[0].endswith(".00") and parts[0] != last_timestamp:
                semantic_events.append(line)
                last_timestamp = parts[0]
                
    return "\n".join(semantic_events)

def extract_ground_truth_and_future_logs(lines):
    category = "Move_Avoid (转移/静止/未判定)"
    future_log =[]
    
    for line in lines:
        line = line.strip()
        if not line: continue
        parts = line.split('|')
        if len(parts) < 2: continue
        try:
            ts = float(parts[0])
            if 20.0 < ts <= 25.0:
                future_log.append(line)
        except ValueError:
            pass

    # 【终极修正】：放弃死板格式匹配，扫描最后 15 行进行关键词模糊宽容嗅探！
    # 防止官方用中文括号（）或把标签放在其他列导致提取失败
    tail_lines = lines[-15:]
    for line in reversed(tail_lines):
        line_str = line.strip()
        if not line_str: continue
        
        # 只要最后这几行包含这些确凿的字眼，直接判定！
        if "救" in line_str or "BeingResuce" in line_str:
            category = "Rescue (救援)"
            break
        elif "雷" in line_str or "Grenade" in line_str or "投掷" in line_str:
            category = "Grenade (丢雷)"
            break
        elif "技能" in line_str or "SkillStart" in line_str:
            category = "Skill (放技能)"
            break
        elif "搜" in line_str or "Looting" in line_str or "物资" in line_str:
            category = "Loot (搜物资)"
            break
        elif "开镜" in line_str or "Action" in line_str:
            category = "Aim (开镜)"
            break
        elif "开火" in line_str or "Fire" in line_str or "伤害" in line_str:
            category = "Fire (开火/伤害)"
            break
            
    future_text = "\n".join(future_log)
    return category, future_text

def main():
    train_dir = "/root/autodl-tmp/决赛训练数据"
    output_file = "/root/autodl-tmp/final_sft_dataset.jsonl"
    teacher_model_path = "/root/autodl-tmp/teacher_model/Qwen2.5-14B-Instruct-AWQ"
    cache_file = "/root/autodl-tmp/stratified_data_cache.pkl"
    
    TARGET_SAMPLES = 10000 
    
    if os.path.exists(cache_file):
        logger.info(f"探测到本地序列化缓存: {cache_file}，极速加载中...")
        with open(cache_file, 'rb') as f:
            stratified_data = pickle.load(f)
    else:
        all_files = glob.glob(os.path.join(train_dir, "**", "*.txt"), recursive=True)
        logger.info(f"挂载原始日志总数: {len(all_files)}")
        
        stratified_data = defaultdict(list)
        try:
            tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Tokenizer 加载异常: {e}")
            return
        
        for f in tqdm(all_files, desc="Fuzzy Tail Sniffing"):
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    
                category, future_5s_log = extract_ground_truth_and_future_logs(lines)
                input_20s = extract_denoised_inputs(lines)
                
                if len(future_5s_log) > 2000:
                    future_5s_log = future_5s_log[:2000] + "\n...[系统拦截：异常海量日志已截断]..."

                raw_prompt = (
                    "你是一名专业的安全巡查AI。请将以下未来5秒的游戏日志提炼为一句规范的中文战术复盘。\n"
                    "【约束条件】\n"
                    f"1. 全局意图锚定：系统已判定该玩家的最终核心决策属于【{category}】，请确保你的描述必须围绕这一意图展开。\n"
                    "2. 格式规范：必须且仅使用“主玩家先...随后...最后...”的结构，明确说明空间位移与战术动作。\n"
                    "3. 词汇白名单：严禁使用“peek、干拉、白给”等非规范口语，强制使用“战术规避、搜寻物资、火力压制”等书面术语。\n"
                    "4. 结构输出：先输出 <thought>...</thought> 分析过程，再输出 <report>...</report> 最终判定。\n"
                    f"【未来5s原始物理日志】\n{future_5s_log}\n"
                )
                
                messages =[{"role": "user", "content": raw_prompt}]
                chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                stratified_data[category].append({
                    "prompt": chat_prompt,
                    "input_20s": input_20s,
                    "filename": os.path.basename(f)
                })
            except Exception:
                continue
                
        logger.info(f"解析完成，持久化至缓存文件: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(stratified_data, f)
            
    logger.info("============== 修复后的目标标签分布基线 ==============")
    total_valid = sum(len(items) for items in stratified_data.values())
    for cat, items in stratified_data.items():
        logger.info(f"- {cat}: {len(items)} 条 ({len(items)/total_valid*100:.2f}%)")
    logger.info("======================================================")
    
    final_selected_data =[]
    random.seed(42)
    
    rare_categories = ["Rescue (救援)", "Grenade (丢雷)", "Skill (放技能)", "Loot (搜物资)", "Fire (开火/伤害)"]
    for cat in rare_categories:
        if cat in stratified_data:
            final_selected_data.extend(stratified_data[cat])
        
    rare_count = len(final_selected_data)
    logger.info(f"长尾高价值特征100%保留率达成，共计: {rare_count} 条")
    
    abundant_categories = ["Aim (开镜)", "Move_Avoid (转移/静止/未判定)"]
    needed_total = TARGET_SAMPLES - rare_count
    
    if needed_total > 0:
        quota_per_cat = needed_total // len(abundant_categories)
        for cat in abundant_categories:
            if cat in stratified_data:
                pool_size = len(stratified_data[cat])
                sample_size = min(pool_size, quota_per_cat)
                logger.info(f"对冗余类别 {cat} 降采样抽取: {sample_size}/{pool_size} 条")
                final_selected_data.extend(random.sample(stratified_data[cat], sample_size))
                
    random.shuffle(final_selected_data)
    logger.info(f"基准真值对齐完成，核心集规模: {len(final_selected_data)} 条")

    valid_prompts =[item["prompt"] for item in final_selected_data]
    valid_inputs_20s =[item["input_20s"] for item in final_selected_data]
    valid_filenames =[item["filename"] for item in final_selected_data]

    if os.path.exists(output_file):
        os.remove(output_file)

    logger.info("实例化 vLLM 计算图调度器...")
    llm = LLM(
        model=teacher_model_path, 
        quantization="awq", 
        tensor_parallel_size=1, 
        # 【显存安全锁死】：降低为0.75，腾出足够空间；并发降为64，杜绝一切OOM可能！
        gpu_memory_utilization=0.75, 
        max_model_len=4096,
        max_num_seqs=64,            
        trust_remote_code=True
    )
    
    sampling_params = SamplingParams(temperature=0.3, max_tokens=300, repetition_penalty=1.1)
    
    chunk_size = 2000
    total_processed = 0
    
    logger.info("执行显存并发知识蒸馏机制...")
    for i in range(0, len(valid_prompts), chunk_size):
        chunk_prompts = valid_prompts[i:i+chunk_size]
        chunk_inputs = valid_inputs_20s[i:i+chunk_size]
        chunk_files = valid_filenames[i:i+chunk_size]
        
        logger.info(f"并发处理 Block: {i} to {i+len(chunk_prompts)} ...")
        outputs = llm.generate(chunk_prompts, sampling_params)
        
        with open(output_file, 'a', encoding='utf-8') as out_f:
            for j, output in enumerate(outputs):
                target_text = output.outputs[0].text.strip()
                if target_text:
                    result = {
                        "file_name": chunk_files[j],
                        "input_text": chunk_inputs[j],
                        "target_text": target_text
                    }
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    total_processed += 1
                    
        logger.info(f"批次写入成功，已累计持久化: {total_processed} 条。")

if __name__ == "__main__":
    main()