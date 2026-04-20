import json
import torch
import os
import multiprocessing
from datasets import load_dataset
# 降维打击：直接使用最稳定原生的 Trainer 和 DataCollator，彻底抛弃不稳定的 SFTTrainer
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

def prepare_instruction_dataset(jsonl_file, tokenizer):
    """
    数据加载与手动分词。我们自己掌控格式化，不依赖外部库。
    """
    print(f"INFO: 正在通过内存映射加载数据集: {jsonl_file}")
    raw_dataset = load_dataset("json", data_files=jsonl_file, split="train")
    
    def format_and_tokenize(example):
        main_map = {0: "交战类", 1: "避战类"}
        sub_map = {
            0: "移动", 1: "被救起", 
            2: "开火", 3: "投掷物", 
            4: "搜刮物资", 5: "施放技能"
        }
        
        text_seq = example.get("nlp_text", "")
        prompt = (
            "你是一个高级游戏行为分析AI。请阅读以下玩家在过去20秒内的对局日志（每秒采样一次核心坐标与事件）：\n"
            f"【对局日志开始】\n{text_seq}\n【对局日志结束】\n"
            "请基于上述日志，预测该玩家在接下来的5秒内，最主要的战术意图（大类：交战类或避战类）以及具体采取的动作（小类：6种细分行为）。\n"
            "请严格按照以下格式输出：\n"
            "意图：[大类结果]\n动作：[小类结果]"
        )
        target = f"意图：{main_map[example['main_label']]}\n动作：{sub_map[example['sub_label']]}"
        
        # 1. 自己拼接完整的问答格式，并在末尾加上结束符 (eos_token)
        full_text = f"{prompt}\n\n【预测结果】\n{target}{tokenizer.eos_token}"
        
        # 2. 直接在这里完成分词，规避 Trainer 的参数冲突
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=1024,
            padding=False # 交给 DataCollator 动态补齐，极大地节省显存
        )
        return tokenized

    num_cpus = os.cpu_count()
    print(f"INFO: 分配 {num_cpus} 个逻辑核心进行文本格式化与分词...")
    
    dataset = raw_dataset.map(
        format_and_tokenize, 
        num_proc=num_cpus,
        remove_columns=raw_dataset.column_names, # 抛弃原始文本，只给模型喂数字ID
        desc="Tokenizing Dataset"
    )
    
    print(f"INFO: 分词处理完成，总样本数: {len(dataset)}")
    return dataset

def train_lora_model():
    model_path = "/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    print(f"INFO: 初始化本地模型与分词器: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 将 tokenizer 传入，直接生成喂给模型的数据集
    full_dataset = prepare_instruction_dataset("nlp_ready_dataset.jsonl", tokenizer)
    dataset_split = full_dataset.train_test_split(test_size=0.05, seed=42)
    
    print("INFO: 配置原生训练超参数...")
    training_args = TrainingArguments(
        output_dir="./game_ai_lora_results",
        per_device_train_batch_size=2,  
        gradient_accumulation_steps=8,  
        gradient_checkpointing=True,          # 【新增显存救星】用少量计算时间换取大量显存空间！
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=1,
        save_strategy="steps",
        save_steps=1000,
        eval_strategy="no",
        bf16=True,
        optim="paged_adamw_32bit",
        report_to="none"
    )
    
    # 调用官方标准的自动对齐整理器
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 【终极防爆】使用最原始、最稳定的 Trainer
    trainer = Trainer(
        model=model,
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
        args=training_args,
        data_collator=data_collator
    )
    
    print("\n---------------------------------------------------------")
    print("INFO: 开始监督式微调 (Native Trainer)")
    print("---------------------------------------------------------\n")
    
    trainer.train()
    
    print("\nINFO: 训练完成，正在保存适配器权重...")
    trainer.model.save_pretrained("./game_ai_final_adapter")
    tokenizer.save_pretrained("./game_ai_final_adapter")
    print("INFO: 任务执行成功。")

if __name__ == "__main__":
    train_lora_model()