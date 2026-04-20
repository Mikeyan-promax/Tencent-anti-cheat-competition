import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import PeftModel, prepare_model_for_kbit_training
from trl import DPOTrainer

# ==========================================
# 阶段二 (下)：大模型生成式特训 - DPO 偏好优化
# 核心目标：从权重根源上教模型抑制“梗语”和“逻辑错误”
# 硬件限制：单卡 RTX 4090 24GB
# ==========================================

def build_dpo_preference_dataset():
    """
    构建 DPO 偏好数据集 (Chosen vs Rejected)
    在实际操作中，这些数据可以基于你 SFT 生成的数据进行人工/规则替换生成。
    这里展示的是符合竞赛大纲精神的核心数据结构。
    """
    print("INFO: 正在构建 DPO 偏好数据集...")
    
    # 构建示范数据：包含 Prompt，好答案(Chosen)，坏答案(Rejected)
    data = {
        "prompt":[
            "你是一名腾讯ACE团队的高级安全巡查Agent。请根据玩家过去20s的降噪日志，推理其未来5s的战术行为。\n【对局日志开始】\n20.00|动作|玩家5374向右移动\n20.00|伤害|玩家5374受击\n【对局日志结束】\n请输出你的思考过程<thought>与最终行为报告<report>。",
            
            "你是一名腾讯ACE团队的高级安全巡查Agent。请根据玩家过去20s的降噪日志，推理其未来5s的战术行为。\n【对局日志开始】\n20.00|射线可见性|发现敌人|15%\n20.00|可搜索的散点物资|破片手榴弹\n【对局日志结束】\n请输出你的思考过程<thought>与最终行为报告<report>。",
            
            "你是一名腾讯ACE团队的高级安全巡查Agent。请根据玩家过去20s的降噪日志，推理其未来5s的战术行为。\n【对局日志开始】\n20.00|玩家基础信息|血量低\n【对局日志结束】\n请输出你的思考过程<thought>与最终行为报告<report>。"
        ],
        "chosen":[
            "<thought>环境分析：玩家受到伤害，且在移动。意图推断：战术规避。</thought><report>主玩家先依托掩体向右侧进行战术规避，随后使用医疗包恢复状态。</report>",
            "<thought>环境分析：视野中出现敌人(15%)，且包内有手雷。意图推断：火力压制。</thought><report>主玩家先发现敌方目标，随后向该方向投掷破片手雷进行排点，最后开镜准备交战。</report>",
            "<thought>状态分析：血量极低。意图推断：避战与恢复。</thought><report>主玩家先迅速脱离交战区域，随后寻找安全位置进行状态恢复。</report>"
        ],
        "rejected":[
            "<thought>玩家被打。跑路。</thought><report>主玩家先大身位干拉，随后被打残白给。</report>", # 含有违禁梗语 (干拉, 白给)
            "<thought>看到人了，扔雷。</thought><report>主玩家先向敌人赛门，随后随便丢个雷。</report>", # 含有违禁梗语 (赛门)
            "<thought>血低。打架。</thought><report>主玩家先被击倒，随后开镜射击。</report>" # 逻辑倒置错误 (击倒后不可能开镜射击)
        ]
    }
    
    dataset = Dataset.from_dict(data)
    # DPO 训练需要拆分 train 和 test
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    return dataset_split

def train_dpo_model():
    """
    执行 DPO 微调，强迫模型对齐优质人类偏好
    加载基础模型与 SFT 适配器，通过 DPOTrainer 抑制“梗语”和“逻辑错误”。
    默认开启终端 tqdm 进度条，实时监控训练进度。
    """
    
    # 这里依然使用你云端的纯净模型路径
    model_path = "/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    print(f"INFO: 初始化 DPO 训练环境，基础模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 【显存防爆第一层】：4-bit NF4 双重量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 加载作为参考的策略基座模型 (Base Model)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    model = prepare_model_for_kbit_training(model)
    
    # 【核心修复】：挂载 SFT 训练好的 LoRA 权重作为基座，而不是初始化全新的 LoRA！
    sft_adapter_path = "./final_seq2seq_adapter"
    if os.path.exists(sft_adapter_path):
        print(f"INFO: 检测到 SFT 适配器，将其挂载以进行 DPO 对齐...")
        # 挂载 SFT 权重，并设置为可训练模式 (is_trainable=True)
        model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=True)
        model.print_trainable_parameters()
    else:
        raise FileNotFoundError(f"找不到 SFT 权重路径 {sft_adapter_path}，请先运行 train_llm_sft_finals.py！")
    
    dataset_split = build_dpo_preference_dataset()
    
    print("INFO: 配置 DPO 专属超参数...")
    training_args = TrainingArguments(
        output_dir="./final_dpo_lora",
        per_device_train_batch_size=1,     # 【显存防爆第三层】：DPO同时加载两个模型(策略和参考)，Batch 必须为 1
        gradient_accumulation_steps=16,    
        gradient_checkpointing=True,       # 【显存防爆第四层】：开启梯度检查点
        learning_rate=5e-5,                # DPO 学习率需比 SFT 更低，防止模型崩坏
        logging_steps=10,
        num_train_epochs=1,                # DPO 对齐只需要极少的轮数
        save_strategy="epoch",
        eval_strategy="no",
        bf16=True,
        optim="paged_adamw_32bit",
        report_to="none",                  # 不使用 wandb，直接在终端打印
        remove_unused_columns=False,       # 配合 trl DPO 必须设为 False
        disable_tqdm=False                 # 强制开启终端 tqdm 进度条
    )
    
    # 初始化 DPOTrainer
    # DPOTrainer 会自动把当前挂载了 SFT LoRA 的模型复制一份，作为冻结的 Reference Model 进行对比计算
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None, # trl 会自动提取带有 peft 的 model 作为 ref_model
        args=training_args,
        beta=0.1, # DPO 的温度参数，控制偏离基座模型的程度。0.1 是标准保守值。
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
        tokenizer=tokenizer,
    )
    
    print("\n---------------------------------------------------------")
    print("INFO: 决赛 4090 显卡特训开始 (DPO 偏好对齐)")
    print("---------------------------------------------------------\n")
    dpo_trainer.train()
    
    print("\nINFO: DPO 对齐完成，正在保存终极适配器权重...")
    dpo_trainer.model.save_pretrained("./final_dpo_adapter")
    tokenizer.save_pretrained("./final_dpo_adapter")
    print("INFO: 恭喜，包含防梗语思维的终极 Agent 模型已就绪！")

if __name__ == "__main__":
    train_dpo_model()