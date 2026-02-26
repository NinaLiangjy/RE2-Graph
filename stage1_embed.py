import json
import numpy as np
import requests
import re
import pickle
from tqdm import tqdm  # 导入进度条库

# --- 配置 ---
VLLM_BASE_URL = "http://localhost:8101/v1"
EMBED_MODEL = "qwen3-8b-embedding"
INPUT_FILE = "your input file.jsonl"
SAVE_PATH = "your output embedding file.pkl"

def clean_tag_content(raw_str):
    # 使用正则彻底移除 <think> 及其内容
    content = re.sub(r'<think>.*?</think>', '', raw_str, flags=re.DOTALL)
    return content.strip()

def stage1_main():
    all_unique_tags = set()
    
    # 1. 解析文件进度
    print("正在解析 JSONL 文件...")
    # 先计算总行数用于显示总进度
    total_lines = sum(1 for _ in open(INPUT_FILE, 'r', encoding='utf-8'))
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        # 使用 tqdm 包裹迭代器
        for line in tqdm(f, total=total_lines, desc="解析标签"):
            try:
                data = json.loads(line)
                clean_str = clean_tag_content(data.get("标签", ""))
                tags = [t.strip() for t in clean_str.replace("\n", ",").replace("，", ",").split(",") if t.strip()]
                all_unique_tags.update(tags)
            except Exception as e:
                continue
    
    unique_tags_list = list(all_unique_tags)
    num_tags = len(unique_tags_list)
    print(f"\n找到唯一标签数: {num_tags}")

    # 2. 调用 vLLM Embedding 进度 (增加分批处理，防止超时)
    print(f"正在生成 Embedding (模型: {EMBED_MODEL})...")
    all_embeddings = []
    batch_size = 756  # 每批处理 128 个标签，可根据显存调整
    
    # 使用 range 配合 tqdm 显示批次进度
    for i in tqdm(range(0, num_tags, batch_size), desc="生成向量"):
        batch_tags = unique_tags_list[i : i + batch_size]
        try:
            response = requests.post(
                f"{VLLM_BASE_URL}/embeddings",
                json={"input": batch_tags, "model": EMBED_MODEL},
                timeout=60  # 设置超时保护
            )
            response.raise_for_status()
            batch_data = response.json()
            # 提取当前批次的 embedding
            batch_vectors = [item["embedding"] for item in batch_data["data"]]
            all_embeddings.extend(batch_vectors)
        except Exception as e:
            print(f"\n批次 {i} 发生错误: {e}")
            # 填充零向量以保持长度一致，或选择中断
            all_embeddings.extend([[0] * 4096] * len(batch_tags))

    embeddings_array = np.array(all_embeddings)

    # 3. 保存结果
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump({"tags": unique_tags_list, "embeddings": embeddings_array}, f)
    print(f"\n✅ 完成！Embedding 已保存至 {SAVE_PATH}, 形状: {embeddings_array.shape}")

if __name__ == "__main__":
    stage1_main()