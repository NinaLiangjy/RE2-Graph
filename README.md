# Refining Event Evolutionary Graphs for Explainable Classification ($\mathrm{RE}^2$-Graph)

## Prerequisites

* **Python**: 3.10+
* **Model Services**: This project relies on local APIs deployed via [vLLM](https://github.com/vllm-project/vllm):
* **LLM API**: `http://localhost:8100` (for tag extraction and normalization)
* **Embedding API**: `http://localhost:8101` (for vectorization)


* **GPU Memory**: A GPU with at least 24GB VRAM (e.g., NVIDIA A10/A800) is recommended to run the Qwen3 series models.

### Installation

```bash
pip install -r requirements.txt

```

---

## Project Structure

| File Name | Phase | Description |
| --- | --- | --- |
| `stage0_extract.py` | **Extraction** | Extracts core event tags from `finance_data.json`. |
| `stage1_embed.py` | **Representation** | Converts tags into high-dimensional embeddings and persists them as `.pkl`. |
| `stage2_cluster_ablationfin.py` | **Transformation** | Clusters tags based on semantic similarity and uses LLM to merge redundant tags. |
| `stage3_graph.py` | **Inference** | Constructs directed/undirected weighted graphs for multi-label event classification and Top-K prediction. |
| `requirements.txt` | **-** | List of required Python dependencies. |

---

## Workflow

### 1. Tag Extraction (Stage 0)

Parses the original complaint narratives to generate an initial tagged dataset.

```bash
python stage0_extract.py

```

*Input: `finance_data.json*`

### 2. Vectorization (Stage 1)

Converts unique tags into semantic vectors to facilitate subsequent similarity calculations.

```bash
python stage1_embed.py

```

*Output: `*.pkl` embedding file.*

### 3. Semantic Clustering & Ablation (Stage 2)

Merges semantically redundant tags by setting different similarity thresholds (e.g., 0.98, 0.95).

```bash
python stage2_cluster_ablationfin.py

```

*Note: You can modify the `SIMILARITY_THRESHOLD` parameter in the script for ablation studies.*

### 4. Graph Construction & Classification (Stage 3 Graph Analysis)

Builds an event evolutionary graph based on the cleaned data and performs department classification tasks.

```bash
python stage3_graph.py --dataset ./dataset/finance_data_tag_0.98.jsonl --wdir ./output --remake

```

* **Performance Evaluation**: Outputs classification accuracy metrics for Top-3, Top-5, and Top-10.


---
