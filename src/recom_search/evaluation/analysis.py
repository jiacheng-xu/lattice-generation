import argparse
import logging
import multiprocessing
from multiprocessing import Pool
import json

import random
import itertools
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import statistics
from collections import defaultdict
from tqdm import tqdm
from src.recom_search.model.util import setup_logger
from src.recom_search.model.model_output import SearchModelOutput
from src.recom_search.evaluation.eval_bench import _get_ngrams, eval_main
from src.recom_search.model.beam_state import BeamNode
from typing import Dict, List
import pickle
from collections import defaultdict
import spacy
nlp = spacy.load("en_core_web_sm")
all_stopwords = spacy.lang.en.stop_words.STOP_WORDS


def branch_facotr(endings):
    nodes_in_len_bucket = [0 for _ in range(30)]
    for key, value in d.items():
        l = value.length
        nodes_in_len_bucket[l] += 1
    nodes_in_len_bucket = [x for x in nodes_in_len_bucket if x != 0]
    nodes_in_len_bucket = nodes_in_len_bucket[:min_len]
    effective_len = len(nodes_in_len_bucket)
    bucket = []
    for i in range(effective_len-1):
        prev, nxt = nodes_in_len_bucket[i], nodes_in_len_bucket[i+1]
        factor = nxt/prev
        bucket.append(factor)
    quants = statistics.quantiles(bucket, n=10)
    pass


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
            logger.warning('self')
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


class Path:
    def __init__(self, tokens=['<s>'], scores=0, ngram_cache=[]) -> None:
        self.tokens = tokens
        self.scores = scores
        self.ngram_cache = ngram_cache
        self.n = 4
    def add(self, tok, score):
        if len(self.tokens) >= self.n-1:
            tmp_ngram = self.tokens[-(self.n-1):] + [tok]
            tmp_ngram = [x.lower().strip() for x in tmp_ngram]
            if tmp_ngram in self.ngram_cache:
                return None
            dup_ngram_cache = self.ngram_cache.copy()
            dup_ngram_cache.append(tmp_ngram)
            obj = Path(self.tokens + [tok], self.scores+score, dup_ngram_cache)
            return obj
        obj = Path(self.tokens + [tok], self.scores+score, [])
        return obj
    def __repr__(self) -> str:
        return self.tokens
    def score(self):
        return self.scores / len(self.tokens) ** 0.8

from heapq import heappop,heappush
def derive_path(nodes: Dict, edges: Dict, eps=int(1e4)):
    # node_uids = [x['uid'] for x in nodes]
    node_text = build_node_text_dict(nodes)
    sos_key, list_of_eos_key, degree_mat = find_start_end(nodes, edges)
    seen = [sos_key]
    edges_tgt_to_src, edges_tgt_to_src_score = cache_edges(edges)

    # dict_path_num = defaultdict(int)    # from sos till uid, how many paths
    # dict_path_num[sos_key] = 1

    paths = defaultdict(list)   #
    paths[sos_key] = [Path()]

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
        h = []
        for x in output:
            # print(x.tokens, x.scores)
            tmp_node = x.add(node_text[node], score)
            if tmp_node == None:
                continue
            heappush(h, (tmp_node.score(), tmp_node))
            if len(h) > eps:
                heappop(h)
        h = [x[1] for x in h]
        paths[node] = h
        # print(node_text[node])
        return h

    for node in list_of_eos_key:
        dfs(node, 0)

    total_path = []
    for end_key in list_of_eos_key:
        path_before_dedup = paths[end_key]

        total_path += paths[end_key]
    return total_path, list_of_eos_key, degree_mat


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


def analyze_graph(paths, nodes):
    # number of paths, number of unique nodes, number of novel ngram, POS tag distributions
    # non-stop word
    stat = {}
    stat['num_path'] = len(paths)
    stat['num_node'] = len(nodes)

    # find non stop word nodes
    nodes_text = [x['text'].strip() for x in nodes.values()]
    trim_nodes_text = [x for x in nodes_text if x not in all_stopwords]
    stat['num_non_stop_node'] = len(trim_nodes_text)
    paths = [[x[0] for x in p] for p in paths]

    all_1grams = [_get_ngrams(1, x) for x in paths]
    flat_1list = list(itertools.chain(*all_1grams))
    uniq_1grams = list(set(flat_1list))
    stat['novel_1gram'] = len(uniq_1grams)

    all_3grams = [_get_ngrams(3, x) for x in paths]
    flat_3list = list(itertools.chain(*all_3grams))
    uniq_3grams = list(set(flat_3list))
    stat['novel_3gram'] = len(uniq_3grams)
    stat['ratio_non_stop'] = len(trim_nodes_text) / len(nodes)
    return stat


def viz_result(generated_outputs: List[BeamNode], ref_sum):
    for go in generated_outputs:
        print(go)
    if len(generated_outputs) == 0:
        return {}
    d_stat = defaultdict(list)
    # net.toggle_stabilization(False)
    # first set all nodes and edges
    all_nodes = {}
    all_edges = {}
    for idx, go in enumerate(generated_outputs):
        nodes, edges = go.visualization()

        all_nodes.update(nodes)
        all_edges.update(edges)
        """
        print(idx)
        paths, degree_mat = derive_path(nodes, edges)
        panda_df, stat = extract_graph_feat(nodes, edges, paths, degree_mat)
        save_dataframe(panda_df, f"{name}_{idx}", "df")
        for k, v in stat.items():
            d_stat[k].append(v)
        """

    all_paths, all_eos, all_degree_mat = derive_path(all_nodes, all_edges)
    # panda_df, all_stat = extract_graph_feat(all_nodes, all_edges, all_paths, all_degree_mat)
    stat = analyze_graph(all_paths, all_nodes)

    abs_degrees = list(all_degree_mat.values())
    stat['degree'] = statistics.mean(abs_degrees)
    random.shuffle(all_paths)
    sampled_paths = all_paths[:50]
    sampled_paths = ["".join([char[0] for char in x]) for x in sampled_paths]
    logger.info(sampled_paths)
    # save_dataframe(panda_df, f"{name}", "df")
    # return d_stat, all_stat
    extrinsic_eval = eval_main(sampled_paths, ref_sum)
    stat = {**stat, **extrinsic_eval}
    logger.info(stat)
    return stat


def test_one_file(f):
    name = ".".join(f.split('.')[:-1])
    config = name.split('_')[2:]
    logger.info(config)
    with open(f"vizs/{f}", 'rb') as fd:
        finished: SearchModelOutput = pickle.load(fd)
    logger.info(f)
    logger.info(finished.ends)
    if not finished.ends:
        return

    stat = viz_result(finished.ends, finished.reference)

    fname = os.path.join('result', "_".join(config)+'.json')
    if os.path.isfile(fname):
        with open(fname, 'r') as read_file:
            data = json.load(read_file)
    else:
        data = []
    stat["file"] = name
    if stat not in data:
        data.append(stat)
    with open(fname, 'w') as wfd:
        json.dump(data, wfd)


if __name__ == "__main__":
    # execute only if run as a script
    files = os.listdir('vizs')
    suffix = '.pkl'
    # suffix = 'bs_10_25_False_0.7_False_False_3_5_0.0_0.9.pkl'
    # suffix = 'dbs_15_35_False_0.7_False_False_3_5_0.0_0.9.pkl'
    # suffix ='recom_sample_15_35_False_0.7_False_False_3_5_0.0_0.9.pkl'
    # suffix='recom_bs_15_35_False_0.7_False_False_3_5_0.0_0.9.pkl'
    # suffix = 'astar_15_35_True_0.5_False_False_3_5_0.0_0.9.pkl'
    # suffix = 'astar_15_35_False_0.7_False_True_3_5_0.5_0.9.pkl'
    suffix = 'astar_15_35_True_0.4_False_False_3_5_True_0.0_0.9.pkl'
    suffix = 'astar_15_35_True_0.4_False_False_4_5_True_0.0_0.9.pkl'
    files = [f for f in files if f.endswith('.pkl') and f.endswith(suffix)]
    f_configs = set(['_'.join(name.split('_')[2:]) for name in files])
    for confi in f_configs:
        print(confi)
        # if 'astar_15_35_True_0.4_False_True_3_5_0.4_0.9' in confi :
        #     pass
        # else:
        #     continue
        logger = setup_logger(name=f"analysis-{confi}")
        f_con = [f for f in files if f.endswith(confi)]
        # test_one_file(files[0])
        # exit()
        test_one_file(files[0])
        with Pool(10) as pool:
            L = pool.map(test_one_file, f_con)
