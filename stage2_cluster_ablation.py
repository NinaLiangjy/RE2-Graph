import pickle
from tqdm import tqdm
import json
import re
import requests
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import faiss
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# --- 配置 ---
LLM_BASE_URL = "http://localhost:8100/v1"
LLM_MODEL = "qwen3-8b"
SAVE_PATH = "your saved embedding file.pkl"
INPUT_FILE = "your input file.jsonl"

# 消融实验参数：在此调整阈值进行多次对比
SIMILARITY_THRESHOLD = 0.98  # 尝试 0.98, 0.95, 0.90 等

def clean_tag_content(raw_str):
    # 使用正则彻底移除 <think> 及其内容
    content = re.sub(r'<think>.*?</think>', '', raw_str, flags=re.DOTALL)
    return content.strip()

def summarize_cluster(tag_group):
    if len(tag_group) <= 1: return tag_group[0]
    
    prompt = f"The following are highly similar tags; please provide the most precise and concise summary term (output the word only):\n{', '.join(tag_group)} /no_think"
    res = requests.post(f"{LLM_BASE_URL}/chat/completions", json={
        "model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0
    })
    return res.json()["choices"][0]["message"]["content"].strip()

def fast_similarity_grouping(embeddings, threshold=0.97):
    """
    使用 Faiss 快速找出余弦相似度 > threshold 的标签组
    """
    threshold = SIMILARITY_THRESHOLD
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)  

    # Faiss 只接受 float32 类型的 numpy 数组
    embeddings = embeddings.astype('float32')
    n, d = embeddings.shape
    # 1. 向量归一化（归一化后的点积等于余弦相似度）
    faiss.normalize_L2(embeddings)  
    # 2. 构建索引 (使用 HNSWFlat 极速搜索，或者简单的 FlatIP)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)   

    # 3. 搜索半径内的邻居
    # 注意：17万数据搜索 0.98 以上的邻居，通常每个点的邻居很少
    print(f"正在进行半径搜索 (r={threshold})...")  
    # 我们搜索每个点最相似的 K 个邻居，或者使用 range_search
    # 这里建议搜 Top 50 足够了，因为 0.98 要求的相似度极高
    # D, I = index.search(embeddings, 50)
    lims, D, I = index.range_search(embeddings, threshold)   

    # 4. 使用并查集或图论构建连通分量
    print("正在构建连通图...")
    G = nx.Graph()
    for i in range(n):
        # 获取第 i 个点的邻居范围
        start, end = int(lims[i]), int(lims[i+1])
        for j_idx in range(start, end):
            neighbor_idx = I[j_idx]
            # 排除自身
            if i != neighbor_idx:
                G.add_edge(i, neighbor_idx)   

    # 获取所有连通分量（即聚类组）
    clusters = list(nx.connected_components(G))   

    # 将结果转换为之前的格式
    labels = np.array([-1] * n)
    for cluster_id, nodes in enumerate(clusters):
        for node in nodes:
            labels[node] = cluster_id           

    # 未被合并的点独立成组
    curr_max = len(clusters)
    for i in range(n):
        if labels[i] == -1:
            labels[i] = curr_max
            curr_max += 1           

    # return labels
    return labels, G

def visualize_clusters(G, tags, tag_mapping, labels, num_clusters_to_show=10):
    # 1. 显著增加画布尺寸 (例如从 12x10 增加到 20x15)
    plt.figure(figsize=(10, 5)) 
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    candidate_clusters = [l for l, c in zip(unique_labels, counts) if c > 1]
    
    # 增加显示的簇数量，以便查看更多结果
    selected_clusters = sorted(candidate_clusters, key=lambda x: np.sum(labels == x), reverse=True)[:num_clusters_to_show]
    
    nodes_to_draw = [i for i, l in enumerate(labels) if l in selected_clusters]
    sub_G = G.subgraph(nodes_to_draw)
    
    # 2. 调整布局算法的参数
    # k 控制节点间的最佳距离。默认通常是 1/sqrt(n)，调大 k (如 0.8 或 1.2) 会让簇之间分得更开
    pos = nx.spring_layout(sub_G, k=1.0, iterations=100, seed=42)
    
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(selected_clusters))) # 使用 tab20 支持更多颜色
    
    for idx, cluster_id in enumerate(selected_clusters):
        cluster_nodes = [n for n in sub_G.nodes() if labels[n] == cluster_id]
        if not cluster_nodes: continue
        
        node_coords = np.array([pos[n] for n in cluster_nodes])
        center = node_coords.mean(axis=0)
        
        # 3. 动态调整气泡半径，防止过大遮挡
        radius = np.max(np.linalg.norm(node_coords - center, axis=1)) + 0.05
        circle = plt.Circle(center, radius, color=colors[idx], alpha=0.1, ec='none')
        plt.gca().add_artist(circle)
        
        # 获取概括词并减小字号
        sample_tag = tags[cluster_nodes[0]]
        summary_word = tag_mapping.get(sample_tag, "Unknown")
        plt.text(center[0], center[1], summary_word, fontsize=10, # 字号稍微调小一点
                 fontweight='bold', color=colors[idx], ha='center', va='center', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # 4. 减小节点大小 (node_size)
    nx.draw_networkx_edges(sub_G, pos, alpha=0.15, edge_color='gray', width=0.5)
    nx.draw_networkx_nodes(sub_G, pos, node_size=20, # 从 30 降到 20
                           node_color=[colors[selected_clusters.index(labels[n])] for n in sub_G.nodes()])
    
    # 5. 确保边距最小化
    plt.tight_layout() 
    plt.axis('off')
    
    save_name = f"cluster_vis_large_{SIMILARITY_THRESHOLD}.svg"
    plt.savefig(save_name, dpi=300, bbox_inches='tight') # bbox_inches='tight' 确保不会切掉边缘内容
    print(f"📊 大尺寸可视化图已保存至: {save_name}")
    plt.show()

def stage2_ablation():
    # 1. 加载 Embedding
    print("正在加载预存的 Embedding 数据...")
    with open(SAVE_PATH, 'rb') as f:
        data = pickle.load(f)
    tags, embeddings = data["tags"], data["embeddings"]

    # 2. 聚类
    print(f"正在进行层次聚类 (阈值: {SIMILARITY_THRESHOLD})...")
    # labels = fast_similarity_grouping(embeddings)
    labels, G = fast_similarity_grouping(embeddings) # 接收 G

    # 3. 概括映射 (加入进度条)
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(tags[idx])
    
    tag_mapping = {}
    cluster_items = list(clusters.values())
    
    print(f"开始调用 LLM 概括 {len(cluster_items)} 个聚类组...")
    # 使用 tqdm 显示 LLM 处理进度
    for group in tqdm(cluster_items, desc="LLM 标签概括"):
        # 如果组内只有一个标签，无需概括
        if len(group) == 1:
            refined = group[0]
        else:
            refined = summarize_cluster(group)
            refined = clean_tag_content(refined)
            
        # 建立旧标签到新标签的映射
        for t in group:
            tag_mapping[t] = refined

    # 4. 生成结果文件 (你已经写好的部分，确保 tqdm 已导入)
    output_name = f"your output file_{SIMILARITY_THRESHOLD}.jsonl"
    print(f"正在更新原始文件并保存至 {output_name}...")
    
    total_lines = sum(1 for _ in open(INPUT_FILE, 'r', encoding='utf-8'))
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(output_name, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=total_lines, desc="写回数据"):
            try:
                item = json.loads(line)
                raw_tag_str = item.get("标签", "")
                
                # 清洗逻辑
                clean_content = re.sub(r'<think>.*?</think>', '', raw_tag_str, flags=re.DOTALL).strip()
                current_tags = [t.strip() for t in clean_content.replace("\n", ",").replace("，", ",").split(",") if t.strip()]
                
                # 转换与去重
                mapped_tags = [tag_mapping.get(t, t) for t in current_tags]
                final_tags = list(dict.fromkeys(mapped_tags))
                
                item["精简标签"] = ",".join(final_tags)
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                
            except Exception as e:
                f_out.write(line)

    print(f"✅ 实验完成！结果已保存至: {output_name}")
    visualize_clusters(G, tags, tag_mapping, labels, num_clusters_to_show=6)
if __name__ == "__main__":
    stage2_ablation()