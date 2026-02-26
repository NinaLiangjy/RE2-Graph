import json
import numpy as np
import requests
import re
import pickle
from tqdm import tqdm
from openai import OpenAI

# --- 配置区 ---
VLLM_API_URL = "http://127.0.0.1:8102/v1"
MODEL_NAME = "qwen3-8b"
INPUT_FILE = "your input path.json"
OUTPUT_FILE = "your output path.jsonl"

# 初始化客户端
client = OpenAI(
    base_url=VLLM_API_URL,
    api_key="token-not-needed",
)

ENGLISH_PROMPT_TEMPLATE = """
Task:
Extract descriptive event tags from financial complaints.

Rules:
1. Extract 2–4 core tags only. Never exceed 4.
2. Each tag must be a concise noun phrase (1–3 words).
3. Each tag must combine:
   - a subject/entity (bank, lender, credit card, account, debt collector, loan, etc.)
   - with an issue/behavior (harassment, misrepresentation, unauthorized charge, billing error, etc.)
4. If multiple distinct problems exist, extract them separately.
5. Do not repeat or paraphrase the same issue.
6. Remove request-oriented language.
7. Use formal financial or compliance terminology only.

Examples:

Complaint:
I have been denied several checking accounts due to identity theft issues. I couldn't pass authentication even with my 4-year address.

Output:
checking account denial, identity theft dispute, authentication failure

Complaint:
I received a letter saying my merchant account was cancelled, but I keep getting emails saying it is still open.

Output:
merchant account status dispute, conflicting account information, account closure inconsistency

Output Requirements (STRICT):
- Output tags only.
- Separate tags using English commas.
- Do NOT include explanations.
- Do NOT include reasoning.
- Do NOT include numbering.
- Do NOT include prefixes such as "Answer:", "Tags:", or any additional text.
- The output MUST begin directly with the first tag.

Complaint:
{content}

Output:
/no_think
"""

def get_tags(content):
    """调用 vLLM API 获取标签并实时返回"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": ENGLISH_PROMPT_TEMPLATE.format(content=content)}
            ],
            temperature=0,
            max_tokens=64
        )
        res_content = response.choices[0].message.content.strip()
        return res_content
    except Exception as e:
        print(f"\n[Error] 请求失败: {e}")
        return "ERROR"

def process_json():
    # 1. 读取原始数据（仍然是 JSON）
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    total = len(data)
    print(f"开始处理，共 {total} 条数据...")

    # 2. 建议在循环外打开输出文件，提高读写效率
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out, \
         tqdm(total=total, desc="整体进度") as pbar:
        
        for item in data:
            desc = item.get("Consumer complaint narrative", "")
            case_id = item.get("id", "unknown")

            # --- 核心修改部分 ---
            if desc and str(desc).strip():  # 确保 desc 不为空且不全是空格
                # 有内容：调用模型抽取
                tags = get_tags(desc)
                item["标签"] = tags
                tqdm.write(f">>> ID: {case_id} | 标签结果: {tags}")
            else:
                # 无内容：跳过调用，标签设为空
                item["标签"] = "" 
                tqdm.write(f">>> ID: {case_id} | 描述为空，跳过抽取")

            # --- 无论是否有描述，都写入结果 ---
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            pbar.update(1)

    print(f"\n所有任务处理完成！最终结果保存在: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_json()
