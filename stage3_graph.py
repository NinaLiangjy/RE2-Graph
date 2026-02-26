# coding=utf-8
from __future__ import unicode_literals
import re
from pyecharts.render import make_snapshot
import networkx as nx
import numpy as np
import pandas as pd
import argparse
import logging
import os
from os.path import join
import collections
from matplotlib import pyplot as plt
import json
import time
from pyecharts.options.series_options import LabelOpts
from tqdm import tqdm
import copy
from pyecharts import options as opts
from pyecharts.charts import Graph, Tab
from pyecharts.commons.utils import JsCode
import math
import random
import addressparser as addr
import spacy
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score,classification_report
# from snapshot_pyppeteer import snapshot


class Reasoning:
    def __init__(self, args):
        self.args = args
        self.wdir = args.wdir
        os.makedirs(self.wdir, exist_ok=True)
        self.logger = logging.getLogger('Reasoning')
        self.logger.addHandler(logging.FileHandler(join(self.wdir, 'out.txt'), mode='a'))
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)
        self.logger.info(time.ctime())
        self.dataset_name = args.dataset.split('/')[-1].replace('.json', '')
        self.data = []
        with open(self.args.dataset, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.data.append(json.loads(line))
                except json.JSONDecodeError:
                    # 打印警告并跳过该行
                    print(f"Warning: Skipping a corrupted JSON line.")
                    continue
        self.data = sorted(self.data, key=lambda x: x['Date received'], reverse=False)
        self.n_data = len(self.data)
        self.train, self.test = self.data[:int(self.n_data * 0.8)], self.data[int(self.n_data * 0.8):]
        self._make_graphs()

        self.nodes_set = {}     # 方便按某种类型遍历顶点
        for node in self.graph.nodes:
            node_type = self.graph.nodes[node]['type']
            if node_type not in self.nodes_set:
                self.nodes_set[node_type] = set()
            self.nodes_set[node_type].add(node)

        print(f'n_nodes: {len(self.graph.nodes)}, n_edges: {len(self.graph.edges)}')    

        for type in self.nodes_set:
            print(f'{type}: {len(self.nodes_set[type])}')

    def _make_graphs(self):
        graph_path = join(self.wdir, 'graph.gexf')
        undirect_graph_path = join(self.wdir, "undirect_graph.gexf")
        if not self.args.remake and os.path.exists(graph_path):
            self.logger.info('从已有数据读取')
            self.graph = nx.read_gexf(graph_path)
            self.undirect_graph = nx.read_gexf(undirect_graph_path)

        else:
            self.logger.info('从原始数据构造')
            data = self.train
            print(f'构图样本数: {len(data)}')

            time1 = time.time()
            n_items = len(data)
            node_2_items = {}
            node_weight = {}
            graph = nx.Graph()
            combined_nodes = {}
            for idx in tqdm(range(n_items), desc='预处理: '):
                row = data[idx]
                for key in row.keys():
                    if row[key] is None:
                        row[key] = f'未标定_{key}'
                label_columns = ['Issue', 'Sub-issue',]
                labels = [row[col] for col in label_columns if row[col] is not None]

                department = row["Sub-product"]
                item_id = idx
                time_str = row['Complaint ID']
                informations_str = row.get('精简标签', "")

                for label in labels:
                    if label not in graph.nodes:
                        graph.add_node(label, type='label')
                        node_2_items[label] = []
                        node_weight[label] = 0

                    node_2_items[label].append(item_id)
                    node_weight[label] += 1

                if department not in graph.nodes:
                    graph.add_node(department, type='department')
                    node_2_items[department] = []
                    node_weight[department] = 0

                node_2_items[department].append(item_id)
                node_weight[department] += 1

                if informations_str:
                    # 使用逗号切分，并去除多余引号和空格
                    # 如果你的逗号可能是中文逗号 '，'，建议使用 .replace('，', ',') 统一一下
                    tag_list = [t.strip().replace('"', '') for t in informations_str.replace('，', ',').split(',')]
                    
                    for tag in tag_list:
                        if not tag: continue  # 跳过空字符串
                        if tag not in graph.nodes:
                            graph.add_node(tag, type='information')
                            node_2_items[tag] = []
                            node_weight[tag] = 0
                            
                        node_2_items[tag].append(item_id)
                        node_weight[tag] += 1

            node_2_items = {k: set(v) for k, v in node_2_items.items()}

            time2 = time.time()
            # 无向图构建
            nodes_list = list(graph.nodes)
            for i, node_i in tqdm(enumerate(nodes_list), desc='无向图构建', total=len(nodes_list)):
                graph.nodes[node_i]['weight'] = node_weight[node_i]
                for node_j in nodes_list[i + 1:]:
                    items_i = node_2_items[node_i]
                    items_j = node_2_items[node_j]
                    co_occurance = len(items_i & items_j)
                    if co_occurance != 0:
                        graph.add_weighted_edges_from([(node_i, node_j, co_occurance)])
                        graph.add_weighted_edges_from([(node_j, node_i, co_occurance)])

            self.undirect_graph = copy.deepcopy(graph)      # 无向图
            nx.write_gexf(self.undirect_graph, undirect_graph_path)

            time3 = time.time()
            # 有向图构建
            digraph = nx.DiGraph()
            digraph.add_nodes_from(graph.nodes(data=True))
            for start_node in tqdm(digraph.nodes, total=len(digraph.nodes), desc='有向图构建'):
                all_weight_sums = 0
                weight_sums = {}
                for end_node in graph.neighbors(start_node):
                    node_type = graph.nodes[end_node]['type']
                    weight = graph.edges[start_node, end_node]['weight']
                    if node_type not in weight_sums:
                        weight_sums[node_type] = 0
                    weight_sums[node_type] += weight
                    all_weight_sums += weight

                for end_node in graph.neighbors(start_node):
                    node_type = graph.nodes[end_node]['type']
                    weight = graph.edges[start_node, end_node]['weight']
                    co_prob = weight / weight_sums[node_type]
                    # co_prob = weight / all_weight_sums
                    digraph.add_weighted_edges_from([(start_node, end_node, co_prob)])

            # self.digraph = digraph

            self.graph = digraph

            time4 = time.time()
            nx.write_gexf(self.graph, graph_path)

            print(time1, time2, time3, time4)
            print(time4-time1)


    
    def classfication(self):
        test_data = self.test
        total_samples = len(test_data)
        print(f"测试集大小: {total_samples}")
        
        # 1. 定义需要统计的 K 值
        ks = [3,5,7,10]
        true_counts = {k: 0 for k in ks}
        

        valid_count = 0  # 实际参与计算的有效样本数
        
        for idx in tqdm(range(total_samples), desc='测试集推理'):
            row = test_data[idx]
            ground_true = row["merge_label" if self.args.merge_label else "Sub-product"]
            
            # 2. 构造查询节点
            query_nodes = []
            for col in ['Issue', 'Sub-issue']:
                val = row.get(col)
                if val: query_nodes.append(str(val).strip())
                
            # 1. 获取标签内容
            tags_data = row.get('精简标签', "")

            # 2. 如果是字符串，先切分为列表；如果是 None 或空，设为空列表
            if isinstance(tags_data, str) and tags_data.strip():
                # 兼容中英文逗号
                raw_tags = [t.strip() for t in tags_data.replace('，', ',').split(',')]
            else:
                raw_tags = []

            # 3. 开始遍历真正的标签列表
            for tag in raw_tags:
                # 去除引号并清理
                tag = tag.replace('\"', '').strip()
                if not tag:
                    continue
                
                # 只有当标签确实在图中时才加入查询列表
                if tag in self.graph.nodes:
                    query_nodes.append(tag)
            
            query_nodes = list(set(query_nodes))

            # 3. 基于图权重进行推理
            departs = {} # {部门名: [权重1, 权重2...]}
            for label in query_nodes:
                if label not in self.graph: continue
                for depart in self.graph[label]:
                    if self.graph.nodes[depart].get('type') == 'department':
                        weight = self.graph[label][depart]["weight"]
                        departs.setdefault(depart, []).append(weight)

            if not departs:
                continue # 如果没有任何匹配部门，跳过该样本
                
            valid_count += 1

            # 4. 计算平均分并排序
            # 结果格式为: [('部门A', 0.85), ('部门B', 0.72), ...]
            infer_res = sorted(
                [(d, sum(w)/len(w)) for d, w in departs.items()],
                key=lambda x: x[1], 
                reverse=True
            )
            
            # 只取部门名称，用于后续判断
            pred_labels = [r[0] for r in infer_res]

            # 5. 核心评估逻辑：命中统计
            for k in ks:
                if ground_true in pred_labels[:k]:
                    true_counts[k] += 1

            # 每 500 条数据打印一次实时进度（可选）
            if valid_count % 500 == 0:
                top3_tmp = true_counts[3] / valid_count
                tqdm.write(f"已处理 {valid_count} 条 | 当前 Top-3 准确率: {top3_tmp:.2%}")

        # 6. 最终报表输出
        print("\n" + "="*40)
        print(f"统计报告 (总样本: {total_samples}, 有效: {valid_count})")
        print("-" * 40)
        for k in ks:
            accuracy = true_counts[k] / valid_count if valid_count > 0 else 0
            print(f"Top-{k:<2} Accuracy (Recall@{k:<2}): {accuracy:.2%}")
        print("="*40)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        default="finance_data_tag_0.94.jsonl",
        help="数据集路径",
    )
    args = parser.parse_args()
    print(f"当前使用的数据集路径为: {args.dataset}")

    if args.remake:
        if os.path.exists(args.wdir):
            import shutil
            shutil.rmtree(args.wdir)

    r = Reasoning(args)
    r.classfication()
