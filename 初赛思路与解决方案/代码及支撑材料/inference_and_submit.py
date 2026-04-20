import os
import glob
import logging
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm  

# Configure logging for academic/engineering standard
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self, base_model_id: str, lora_checkpoint_path: str):
        self.base_model_id = base_model_id
        self.lora_checkpoint_path = lora_checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing Inference Engine on {self.device.upper()}...")

        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # 【关键修复 1】：强制添加 local_files_only=True，杜绝联网卡死
        logger.info(f"Loading tokenizer locally from {base_model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, 
            trust_remote_code=True,
            local_files_only=True 
        )
        
        # 【关键修复 2】：强制添加 local_files_only=True
        logger.info(f"Loading base model locally from {base_model_id} with 4-bit quantization...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, 
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        
        logger.info(f"Applying PEFT LoRA adapters from {self.lora_checkpoint_path}...")
        if not os.path.exists(self.lora_checkpoint_path):
            raise FileNotFoundError(f"LoRA checkpoint directory not found: {self.lora_checkpoint_path}")
            
        self.model = PeftModel.from_pretrained(base_model, self.lora_checkpoint_path)
        self.model.eval()
        logger.info("Model evaluation mode activated. Ready for inference.")

    def preprocess_sequence(self, txt_path: str) -> str:
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Failed to read file {txt_path}: {e}")
            return ""

        semantic_events = []
        last_timestamp = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('|')
            if len(parts) < 2:
                continue
                
            timestamp = parts[0]
            event_type = parts[1]
            
            if event_type in ["游戏开始", "动作", "伤害", "技能", "可搜索的散点物资"]:
                semantic_events.append(line)
            elif event_type == "玩家基础信息":
                if timestamp.endswith(".00") and timestamp != last_timestamp:
                    semantic_events.append(line)
                    last_timestamp = timestamp
                    
        return "\n".join(semantic_events)

    def generate_prediction(self, clean_text: str) -> tuple:
        prompt = (
            "你是一个高级游戏行为分析AI。请阅读以下玩家在过去20秒内的对局日志（每秒采样一次核心坐标与事件）：\n"
            f"【对局日志开始】\n{clean_text}\n【对局日志结束】\n"
            "请基于上述日志，预测该玩家在接下来的5秒内，最主要的战术意图（大类：交战类或避战类）以及具体采取的动作（小类：6种细分行为）。\n"
            "请严格按照以下格式输出：\n"
            "意图：[大类结果]\n动作：[小类结果]"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=50, 
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False 
            )
            
        input_length = inputs['input_ids'].shape[1]
        generated_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        return self._parse_labels(generated_text)

    def _parse_labels(self, generated_text: str) -> tuple:
        main_label_str = ""
        sub_label_str = ""
        
        for line in generated_text.split('\n'):
            if "意图" in line:
                main_label_str = line
            elif "动作" in line:
                sub_label_str = line
                
        main_val = 0 if "交战" in main_label_str else 1
        
        sub_val = 0
        if "救" in sub_label_str or "BeingResuce" in sub_label_str:
            sub_val = 1
        elif "火" in sub_label_str or "Fire" in sub_label_str:
            sub_val = 2
        elif "投掷" in sub_label_str or "雷" in sub_label_str or "Grenade" in sub_label_str:
            sub_val = 3
        elif "搜" in sub_label_str or "物资" in sub_label_str or "Looting" in sub_label_str:
            sub_val = 4
        elif "技能" in sub_label_str or "Skill" in sub_label_str:
            sub_val = 5
            
        return main_val, sub_val

def execute_batch_inference():
   # 使用包含乱码的绝对路径，确保一定能读到
    test_dir = "/root/autodl-tmp/测试1000题/#U6d4b#U8bd51000#U9898"
    output_excel_path = "Final_Submission.xlsx"
    
    if not os.path.exists(test_dir):
        logger.error(f"Test directory '{test_dir}' not found. Please extract the test dataset.")
        return

    try:
        engine = InferenceEngine(
            # 【关键修复 3】：替换为你服务器上的真实绝对路径
            base_model_id="/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
            lora_checkpoint_path="./game_ai_lora_results/checkpoint-7000"
        )
    except Exception as e:
        logger.error(f"Failed to initialize Inference Engine: {e}")
        return
    
    test_files = glob.glob(os.path.join(test_dir, "*.txt"))
    test_files.sort()
    
    total_files = len(test_files)
    logger.info(f"Found {total_files} test files. Commencing batch inference...")
    
    results_list = []
    progress_bar = tqdm(test_files, desc="Inferencing", unit="seq")
    
    for file_path in progress_bar:
        file_name = os.path.basename(file_path)
        
        try:
            question_id = int(file_name.replace(".txt", ""))
        except ValueError:
            logger.warning(f"Unexpected filename format: {file_name}. Skipping.")
            continue
            
        clean_text = engine.preprocess_sequence(file_path)
        main_class, sub_class = engine.generate_prediction(clean_text)
        
        results_list.append({
            "题目序号": question_id,
            "意图决策": main_class,
            "动作行为": sub_class
        })
        
        progress_bar.set_postfix({
            "Q_ID": question_id,
            "Intent": main_class, 
            "Action": sub_class
        })
            
    logger.info("Inference complete. Constructing pandas DataFrame and exporting to Excel format...")
    df_submission = pd.DataFrame(results_list)
    df_submission = df_submission.sort_values(by="题目序号").reset_index(drop=True)
    
    try:
        df_submission.to_excel(output_excel_path, index=False)
        logger.info(f"Successfully generated official submission file: {output_excel_path}")
    except Exception as e:
        logger.error(f"Failed to export Excel file: {e}")

if __name__ == "__main__":
    execute_batch_inference()