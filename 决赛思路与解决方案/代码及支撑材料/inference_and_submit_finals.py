import os
import glob
import logging
import pandas as pd
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm  

# 规范化学术级日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalsInferenceEngine:
    def __init__(self, base_model_id: str, lora_checkpoint_path: str):
        """
        初始化端侧离线推理引擎
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"初始化推理引擎，当前计算设备: {self.device.upper()}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_id, trust_remote_code=True, local_files_only=True 
        )
        self.tokenizer.padding_side = 'left' # 推理时采用左侧填充更利于生成
        
        # 物理级拦截黑名单
        bad_words =["peek", "干拉", "大身位", "白给", "赛门", "刘涛"]
        self.bad_words_ids =[]
        for word in bad_words:
            ids = self.tokenizer.encode(word, add_special_tokens=False)
            if ids:
                self.bad_words_ids.append(ids)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True, local_files_only=True
        )
        
        logger.info(f"加载端侧特定域适配器：{lora_checkpoint_path}")
        self.model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
        self.model.eval()

    def preprocess_20s_sequence(self, txt_path: str) -> str:
        """输入特征降噪预处理"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"IO 读取异常 {txt_path}: {e}")
            return ""

        semantic_events =[]
        last_timestamp = ""
        for line in lines:
            line = line.strip()
            if not line: continue
            parts = line.split('|')
            if len(parts) < 2: continue
            
            timestamp, event_type = parts[0], parts[1]
            if event_type in ["游戏开始", "动作", "伤害", "技能", "可搜索的散点物资"]:
                semantic_events.append(line)
            elif event_type == "玩家基础信息":
                if timestamp.endswith(".00") and timestamp != last_timestamp:
                    semantic_events.append(line)
                    last_timestamp = timestamp
        return "\n".join(semantic_events)

    def heuristic_fallback_check(self, clean_text: str, generated_report: str) -> str:
        """
        启发式特征兜底校验 (Heuristic Fallback)
        """
        # 校验 1：无源投掷幻觉拦截
        if "手雷" in generated_report or "丢雷" in generated_report or "破片" in generated_report:
            if "破片手榴弹" not in clean_text and "搜刮" not in clean_text and "物资" not in clean_text:
                logger.warning("触发边界异常规则：检测到无源投掷幻觉，执行语义降级替换。")
                generated_report = generated_report.replace("向敌方目标投掷破片手雷", "进行战术空间拉扯")
                generated_report = generated_report.replace("投掷破片手雷", "走位规避")
                generated_report = generated_report.replace("投掷手雷", "战术走位")
                generated_report = generated_report.replace("丢雷", "规避")
                
        return generated_report

    def generate_seq2seq_prediction(self, clean_text: str) -> str:
        """基于 Agentic CoT 的长文本行为序列生成"""
        
        # 时序长度阈值截断保护 (对齐 SFT 阶段机制)
        if len(clean_text) > 2500:
            clean_text = "...[早期日志已省略]...\n" + clean_text[-2500:]

        # 【核心修正】：补齐 SFT 阶段的专属触发标识符
        prompt = (
            "你是一名腾讯ACE团队的高级安全巡查Agent。请根据玩家过去20s的降噪日志，推理其未来5s的战术行为。注意环境视野（射线可见性）与自身状态。\n"
            f"【对局日志开始】\n{clean_text}\n【对局日志结束】\n"
            "请输出你的思考过程<thought>与最终行为报告<report>。"
            "\n\n### 预测结果:\n"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=350,               # 确保有足够的输出空间包含 thought 和 report
                pad_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=self.bad_words_ids, # 物理级阻断违禁梗语
                temperature=0.2,                  # 极低温度，保证因果推理的确定性
                repetition_penalty=1.15,          
                do_sample=True 
            )
            
        input_length = inputs['input_ids'].shape[1]
        generated_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        # 正则提取 Report 区块
        match = re.search(r"<report>(.*?)</report>", generated_text, re.DOTALL)
        if match:
            report_text = match.group(1).strip()
        else:
            report_text = generated_text.replace("<thought>", "").replace("</thought>", "").strip()
            
        safe_report = self.heuristic_fallback_check(clean_text, report_text)
        return safe_report

def execute_finals_inference():
    test_dir = "/root/autodl-tmp/决赛测试100题"
    output_excel_path = "/root/autodl-tmp/Final_Submission_5s.xlsx"
    
    engine = FinalsInferenceEngine(
        base_model_id="/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
        lora_checkpoint_path="./final_seq2seq_adapter" 
    )
    
    test_files = glob.glob(os.path.join(test_dir, "**", "*.txt"), recursive=True)
    logger.info(f"成功解析测试集文件数: {len(test_files)}，执行端侧推理流水线...")
    
    results_list =[]
    for file_path in tqdm(test_files, desc="Agentic CoT Inferencing"):
        file_name = os.path.basename(file_path)
        try:
            question_id = int(re.sub(r'\D', '', file_name))
        except ValueError:
            continue
            
        clean_text = engine.preprocess_20s_sequence(file_path)
        final_report = engine.generate_seq2seq_prediction(clean_text)
        
        results_list.append({
            "题目序号": question_id,
            "后5秒续写": final_report
        })
            
    df_submission = pd.DataFrame(results_list)
    df_submission = df_submission.sort_values(by="题目序号").reset_index(drop=True)
    
    try:
        df_submission.to_excel(output_excel_path, index=False)
        logger.info(f"预测结果成功输出至目标矩阵路径: {output_excel_path}")
    except Exception as e:
        logger.error(f"导出 Excel 矩阵文件失败: {e}")

if __name__ == "__main__":
    execute_finals_inference()