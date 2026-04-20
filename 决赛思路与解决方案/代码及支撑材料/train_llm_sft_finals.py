import torch
import os
import logging
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling  # 【修复】：使用最底层、绝对不会报错的原生数据整理器
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# 规范化系统日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_seq2seq_dataset(jsonl_file, tokenizer):
    logger.info(f"加载高质量 Agentic CoT 数据集: {jsonl_file}")
    raw_dataset = load_dataset("json", data_files=jsonl_file, split="train")
    
    max_seq_len = 4096 
    
    def format_and_tokenize(example):
        input_text = example.get("input_text", "")
        target_text = example.get("target_text", "")
        
        # 【终极防截断保护 & 提速黑科技】
        # 正常的20秒日志约1000字符。如果异常庞大（超过2500字符），说明是外挂级高频操作。
        # 我们直接截取字符串的“后 2500 个字符”（最靠近未来的关键因果动作）
        if len(input_text) > 2500:
            input_text = "...[早期日志已省略]...\n" + input_text[-2500:]
            
        prompt = (
            "你是一名腾讯ACE团队的高级安全巡查Agent。请根据玩家过去20s的降噪日志，推理其未来5s的战术行为。注意环境视野（射线可见性）与自身状态。\n"
            f"【对局日志开始】\n{input_text}\n【对局日志结束】\n"
            "请输出你的思考过程<thought>与最终行为报告<report>。"
        )
        
        response_template = "\n\n### 预测结果:\n"
        full_text = f"{prompt}{response_template}{target_text}{tokenizer.eos_token}"
        
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_len, 
            padding=False 
        )
        
        tokenized["length"] = len(tokenized["input_ids"])
        return tokenized

    num_cpus = os.cpu_count()
    dataset = raw_dataset.map(
        format_and_tokenize, 
        num_proc=num_cpus,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing Dataset"
    )
    
    lengths = dataset["length"]
    max_len_found = max(lengths)
    truncated_count = sum(1 for l in lengths if l >= max_seq_len)
    
    logger.info(f"数据集分词完成。样本总数: {len(dataset)}")
    logger.info(f"当前数据集中最大 Token 长度为: {max_len_found}")
    if truncated_count > 0:
        logger.warning(f"[warning]警告: 仍有 {truncated_count} 条数据被强行截断！")
    else:
        logger.info(f"[info]完美: 所有数据均在 {max_seq_len} 安全长度内，零截断，目标文本 100% 完整保留！")

    return dataset

def train_seq2seq_model():
    model_path = "/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    logger.info(f"挂载预训练基座模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True, local_files_only=True
    )
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32, 
        lora_alpha=64, 
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    full_dataset = prepare_seq2seq_dataset("/root/autodl-tmp/final_sft_dataset.jsonl", tokenizer)
    dataset_split = full_dataset.train_test_split(test_size=0.05, seed=42)
    
    logger.info("构建带有 TensorBoard 监控与早停机制的训练管线...")
    training_args = TrainingArguments(
        output_dir="./final_seq2seq_lora", 
        per_device_train_batch_size=1,     
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,    
        gradient_checkpointing=True,       
        learning_rate=2e-4,
        num_train_epochs=2,                
        
        logging_dir="./tf-logs",           
        logging_steps=10,                  
        report_to="tensorboard",           
        eval_strategy="steps",             
        eval_steps=50,                     
        save_strategy="steps",             
        save_steps=50,                     
        save_total_limit=3,                
        load_best_model_at_end=True,       
        metric_for_best_model="eval_loss", 
        greater_is_better=False,           
        
        bf16=True,
        optim="paged_adamw_32bit"
    )
    
    # 【修复】：使用原生语言模型整理器，兼容性 100%，绝不报错！
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
        args=training_args,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
    )
    
    logger.info("=========================================================")
    logger.info("决赛 4090 显卡训练开始 (支持 TensorBoard 实况与早停机制)")
    logger.info("=========================================================")
    trainer.train()
    
    logger.info("微调终止，正在持久化最优泛化权重...")
    trainer.model.save_pretrained("./final_seq2seq_adapter")
    tokenizer.save_pretrained("./final_seq2seq_adapter")
    logger.info("最优权重落盘完成，模型部署就绪。")

if __name__ == "__main__":
    train_seq2seq_model()