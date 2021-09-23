from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

import statistics
from collections import defaultdict
from tqdm import tqdm
from src.recom_search.model.beam_state import BeamNode
from typing import Dict, List
import pickle
from collections import defaultdict


def find_start_end(nodes, edges):
    degree = {}
    out_degree = {}
    for node in nodes.values():
        degree[node['uid']] = 0
        out_degree[node['uid']] = 0
    for edge in edges.values():
        degree[edge['tgt']] += 1
        out_degree[edge['src']] += 1
    key_of_sos = [k for k, v in degree.items() if v == 0]
    key_of_eos = [k for k, v in out_degree.items() if v == 0]
    assert len(key_of_sos) == 1
    return key_of_sos[0], key_of_eos, degree


def cache_edges(edges):
    edge_info = defaultdict(list)
    edge_score = defaultdict(list)
    for edge in edges.values():
        if edge['tgt'] == edge['src']:
            print('self')
            continue
        edge_info[edge['tgt']].append(edge['src'])
        edge_score[edge['tgt']].append(edge['score'])
    return edge_info, edge_score


def build_node_text_dict(nodes):
    d = {}
    for n in nodes.values():
        d[n['uid']] = n['text']
    return d


def avg_score(path):
    return statistics.mean([x[1] for x in path])


def sum_score(path):
    return sum([x[1] for x in path])


def ext_text(path):
    return "".join([x[0] for x in path])


def reward_len(path, alpha=0.01):
    return len(path) * alpha


def derive_path(nodes: Dict, edges: Dict):
    # node_uids = [x['uid'] for x in nodes]
    node_text = build_node_text_dict(nodes)
    sos_key, list_of_eos_key, degree_mat = find_start_end(nodes, edges)
    seen = [sos_key]
    edges_tgt_to_src, edges_tgt_to_src_score = cache_edges(edges)

    # dict_path_num = defaultdict(int)    # from sos till uid, how many paths
    # dict_path_num[sos_key] = 1

    paths = defaultdict(list)   #
    paths[sos_key] = [[('<s>', 0)]]

    def dfs(node, score):
        # print(node)
        if node in seen:
            return paths[node]

        fathers = edges_tgt_to_src[node]
        fathers_score = edges_tgt_to_src_score[node]
        output = []
        for f, fs in zip(fathers, fathers_score,):
            output += dfs(f, fs)
        seen.append(node)
        output = [x + [(node_text[node], score)] for x in output]
        paths[node] = output
        return output

    for node in list_of_eos_key:
        dfs(node, 0)

    total_path = []
    for end_key in list_of_eos_key:
        total_path += paths[end_key]
    return total_path, degree_mat


# things we care about: distribution of the path's score
# len of paths
# in-degree of nodes


def extract_graph_feat(nodes, edges, paths, node_degree):
    stat = {}
    # number of unique nodes
    stat['num_nodes'] = len(nodes)
    stat['num_edges'] = len(edges)
    stat['num_paths'] = len(paths)

    # analyze each path
    data = []
    for p in paths:
        avg_s = avg_score(p)
        sum_s = sum_score(p)
        len_path = len(p)
        txt = ext_text(p)
        data.append([avg_s, sum_s, len_path, txt])
    data = np.array(data)
    df = pd.DataFrame.from_records(data, columns=['avg', 'sum', 'len', 'txt'])
    return df, stat


def save_dataframe(df, fname, path):
    with open(os.path.join(path, fname+'.pkl'), 'wb') as fd:
        pickle.dump(df, fd)

def analyze_graph(nodes, edges):
    pass

def viz_result(generated_outputs: List[BeamNode], name):
    for go in generated_outputs:
        print(go)
    d_stat = defaultdict(list)
    # net.toggle_stabilization(False)
    # first set all nodes and edges
    all_nodes = {}
    all_edges = {}
    for idx, go in enumerate(generated_outputs):
        nodes, edges = go.visualization()
        all_nodes.update(nodes)
        all_edges.update(edges)
        print(idx)
        paths, degree_mat = derive_path(nodes, edges)
        panda_df, stat = extract_graph_feat(nodes, edges, paths, degree_mat)
        save_dataframe(panda_df, f"{name}_{idx}", "df")
        for k, v in stat.items():
            d_stat[k].append(v)

    all_paths, all_degree_mat = derive_path(all_nodes, all_edges)
    panda_df, all_stat = extract_graph_feat(
        all_nodes, all_edges, all_paths, all_degree_mat)
    save_dataframe(panda_df, f"{name}", "df")
    return d_stat, all_stat


if __name__ == "__main__":
    # execute only if run as a script
    files = os.listdir('vizs')
    files = [f for f in files if f.endswith('.pkl') and f.startswith('best')]
    d_stat = defaultdict(list)
    d_stat_all = defaultdict(list)
    import random
    # random.shuffle(files)
    
    for f in tqdm(files):
        name = f.split('.')[0]

        with open(f"vizs/{f}", 'rb') as fd:
            finished = pickle.load(fd)
        print(f)
        stat_single, stat_all = viz_result(finished, name)
        
        for k in stat_single.keys():
            if len(stat_single[k]) <= 1:
                continue
            print(f"Key:{k}")
            print(statistics.quantiles(stat_single[k]))
            print(f"Full {k}: {stat_all[k]}")
            print('-'*10)
        print('\n\n')
        for k, v in stat_single.items():
            d_stat[k] += v
        for k, v in stat_all.items():
            d_stat_all[k].append(v)


    