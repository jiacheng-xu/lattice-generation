from pyvis.network import Network
from rouge_score import rouge_scorer

from heapq import heappop, heappush

import logging

logger = logging.getLogger(__name__)

from src.recom_search.evaluation.vis import draw_edges, draw_nodes
from src.recom_search.model.setup import tokenizer
import argparse

import multiprocessing
from multiprocessing import Pool
import json
from filelock import FileLock
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
from src.recom_search.model.setup import setup_logger
from src.recom_search.model.model_output import SearchModelOutput
from src.recom_search.evaluation.eval_bench import _get_ngrams, eval_main,bleu_scorer, group_bleu, self_bleu, self_edit_distance
from src.recom_search.model.beam_state import BeamNode
from typing import Dict, List
import pickle
from collections import defaultdict
# import spacy

# nlp = spacy.load("en_core_web_sm")
# all_stopwords = spacy.lang.en.stop_words.STOP_WORDS



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
    d_tok_id = {}
    for n in nodes.values():
        d[n['uid']] = n['text']
        d_tok_id[n['uid']] = n['tok_idx']
    return d, d_tok_id


def avg_score(path):
    return statistics.mean([x[1] for x in path])


def sum_score(path):
    return sum([x[1] for x in path])


def ext_text(path):
    return "".join([x[0] for x in path])


def reward_len(path, alpha=0.01):
    return len(path) * alpha


scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=False)


class GenPath:
    def __init__(self, tokens=['<s>'], token_ids=[2], scores=0, ngram_cache=[],max_len=200,dedup_n=None) -> None:
        self.tokens = tokens
        self.token_ids = token_ids
        self.scores = scores
        self.ngram_cache = ngram_cache
        self.n = dedup_n
        self.metrics = {}
        self.max_len =max_len 


    def add(self, tok, tok_id, score):
        if self.n and len(self.tokens) >= self.n-1 :
            tmp_ngram = self.tokens[-(self.n-1):] + [tok]
            tmp_ngram = [x.lower().strip() for x in tmp_ngram]
            if tmp_ngram in self.ngram_cache or len(self.tokens) > self.max_len:
                return None
            dup_ngram_cache = self.ngram_cache.copy()
            dup_ngram_cache.append(tmp_ngram)
            obj = GenPath(self.tokens + [tok], self.token_ids +
                       [tok_id], self.scores+score, dup_ngram_cache, max_len=self.max_len,dedup_n=self.n)
            return obj
        obj = GenPath(self.tokens + [tok], self.token_ids +
                   [tok_id], self.scores+score, [], max_len=self.max_len,dedup_n=self.n)
        return obj

    def __len__(self) -> int:
        return len(self.tokens)

    def __repr__(self) -> str:
        return self.tokens

    def score(self):
        return self.scores / len(self.tokens) ** 0.8


def derive_path(nodes: Dict, edges: Dict,flag_sum:bool, eps=int(5e3), min_len=5):
    # node_uids = [x['uid'] for x in nodes]
    node_text, node_tok_idx = build_node_text_dict(nodes)
    sos_key, list_of_eos_key, degree_mat = find_start_end(nodes, edges)



    seen = [sos_key]
    edges_tgt_to_src, edges_tgt_to_src_score = cache_edges(edges)
    if flag_sum:
        dedup = 4
        max_len = 50
    else:
        dedup = None
        max_len = 200
    # dict_path_num = defaultdict(int)    # from sos till uid, how many paths
    # dict_path_num[sos_key] = 1

    paths = defaultdict(list)   #
    paths[sos_key] = [GenPath(dedup_n=dedup,max_len=max_len)]

    def dfs(node, score):
        # print(node)
        if node in seen:
            return paths[node]

        fathers = edges_tgt_to_src[node]
        fathers_score = edges_tgt_to_src_score[node]
        output = []
        for f, fs in zip(fathers, fathers_score):
            output += dfs(f, fs)
        seen.append(node)

        # filter_output = [x.add(node_text[node], score) for x in output]
        filter_output = list(
            map(lambda x: x.add(node_text[node], node_tok_idx[node], score), output))
        filter_output = list(filter(lambda x: x != None, filter_output))
        random.shuffle(filter_output)
        filter_output = filter_output[:eps]

        paths[node] = filter_output
        # print(node_text[node])
        return filter_output

    for node in list_of_eos_key:
        dfs(node, 0)

    total_path = []
    for end_key in list_of_eos_key:
        # path_before_dedup = paths[end_key]
        # deduplication already happens in Path class
        available_paths = paths[end_key]
        available_paths = [x for x in available_paths if len(
            x) >= min_len]
        total_path += available_paths
    return total_path, list_of_eos_key, degree_mat, sos_key


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


def analyze_graph(paths: List[GenPath], nodes):
    # number of paths, number of unique nodes, number of novel ngram, POS tag distributions
    # non-stop word
    stat = {}
    stat['num_path'] = len(paths)
    stat['num_node'] = len(nodes)

    # find non stop word nodes
    # nodes_text = [x['text'].strip() for x in nodes.values()]
    # trim_nodes_text = [x for x in nodes_text if x not in all_stopwords]
    # stat['num_non_stop_node'] = len(trim_nodes_text)
    paths = [p.tokens for p in paths]

    all_1grams = [_get_ngrams(1, x) for x in paths]
    flat_1list = list(itertools.chain(*all_1grams))
    uniq_1grams = list(set(flat_1list))
    stat['novel_1gram'] = len(uniq_1grams)


    all_2grams = [_get_ngrams(2, x) for x in paths]
    flat_2list = list(itertools.chain(*all_2grams))
    uniq_2grams = list(set(flat_2list))
    stat['novel_2gram'] = len(uniq_2grams)
    all_3grams = [_get_ngrams(3, x) for x in paths]
    flat_3list = list(itertools.chain(*all_3grams))
    uniq_3grams = list(set(flat_3list))
    stat['novel_3gram'] = len(uniq_3grams)
    # stat['ratio_non_stop'] = len(trim_nodes_text) / len(nodes)
    return stat


def _evaluate_rouge(path_obj: GenPath, ref):
    decoded_text = tokenizer.decode(
        path_obj.token_ids, skip_special_tokens=True)
    # if random.random() < 0.002:
    #     logging.info(f"Comparing {decoded_text} with {ref}")
    s = scorer.score(decoded_text, ref)
    f2 = s['rouge2'].fmeasure

    path_obj.metrics['rouge2'] = f2
    path_obj.ngram_cache = None
    path_obj.text = decoded_text


def _evaluate_bleu(path_obj: GenPath, ref):
    decoded_text = tokenizer.decode(
        path_obj.token_ids, skip_special_tokens=True)
    if random.random() < 0.01:
        logging.info(f"Comparing {decoded_text} with {ref}")
    s = bleu_scorer.sentence_score(decoded_text, [ref])
    path_obj.metrics['bleu'] = s.score
    path_obj.ngram_cache = None
    path_obj.text = decoded_text


def oracle_path(cand_paths: List, ref_sum, flag_sum, oracle_samples=20):
    if flag_sum:
        f = _evaluate_rouge
    else:
        f = _evaluate_bleu
    random.shuffle(cand_paths)
    original_len = len(cand_paths)
    cand_paths = cand_paths[:10000]
    logging.info(f"Cutting {original_len} to {len(cand_paths)}")
    list(map(lambda x: f(x, ref_sum), cand_paths))
    logging.info(f"Done with calculating ROUGE/BLEU")
    if flag_sum:
        cand_paths.sort(key=lambda p: p.metrics['rouge2'], reverse=True)
    else:
        cand_paths.sort(key=lambda p: p.metrics['bleu'], reverse=True)
    logging.info(f"Done with sorting")
    var_paths = defaultdict(list)
    oracle_sents = cand_paths[:oracle_samples]
    var_paths[f"oracle_{oracle_samples}"] = oracle_sents
    oracle_top1 = cand_paths[:1]
    var_paths['oracle_1'] = oracle_top1

    sampled_paths = random.choices(cand_paths, k=oracle_samples*10)
    var_paths['sample'] = sampled_paths
    
    # oracle_sents = [ (x.text, len(x), x.metrics) for x in path_oracle]
    # bucket
    bucket = [10, 20 , 40, 1000]
    for samp_path in sampled_paths:
        l = len(samp_path)

        for idx, b in enumerate(bucket):
            if l<=b:
                var_paths[f"buck_{b}"].append(samp_path)
                break

    stat = {}
    for idx, b in enumerate(bucket):
        rate = len(var_paths[f"buck_{b}"]) / len(var_paths['sample'])
        stat[f"ratio_{b}"] = rate
    return var_paths, stat

def random_walk_from_sos(sos_key, all_nodes, all_edges, max_len=100):
    # edges[f"{p_uid}_{node_uid}"] = edge_info
    # nodes[node.uid] = {   'uid': node_uid,
                # 'text': node.token_str,
                # 'tok_idx': node.token_idx
    # get the chain of node id
    rt = [sos_key]
    edges = list(all_edges.keys())
    while True:
        ele = rt[-1]
        cand_edges = [ e.split('_')[1] for e in edges if e.startswith(ele)]
        if not cand_edges:
            break
        rand_nxt  =random.choice(cand_edges)
        rt.append(rand_nxt)
        if len(rt) >= max_len:
            break
    rt_token_ids = [all_nodes[uid]['tok_idx'] for uid in rt]

    return rt_token_ids

def get_random_walk_eval(sos_key,nodes,edges, nsample=5):
    group = []
    cnt = 0
    while cnt <=nsample:
        rand_1 = random_walk_from_sos(sos_key,nodes,edges)
        sample_sent1 = tokenizer.decode(rand_1, skip_special_tokens=True)
        group.append(sample_sent1)
        cnt += 1
    stat = {}
    b =  self_bleu(group)
    ed_dis = self_edit_distance(group)
    stat['rand_walk_self_bleu'] = b
    stat['rand_walk_self_edit'] = ed_dis
    return stat
def viz_result(generated_outputs: List[BeamNode], ref_sum, flag_sum, nsample=20):
    logging.info('\n\n---')
    for go in generated_outputs:
        logging.info(go)
    if len(generated_outputs) == 0:
        return {}
    
    # first set all nodes and edges
    all_nodes = {}
    all_edges = {}
    net = Network(height='1500px', width='100%', directed=True)

    for idx, go in enumerate(generated_outputs):
        nodes, edges = go.visualization()
        all_nodes.update(nodes)
        all_edges.update(edges)
        draw_nodes(net, nodes, idx)
        draw_edges(net, edges, idx)

    all_paths, all_eos, all_degree_mat,sos_key = derive_path(all_nodes, all_edges, flag_sum)
    
    stat = analyze_graph(all_paths, all_nodes)

    abs_degrees = list(all_degree_mat.values())
    stat['degree'] = statistics.mean(abs_degrees)

    # compute ROUGE or BLEU
    logging.info("ALL path")
    logging.info(len(all_paths))
    dict_of_var_paths ,buck_ratio =  oracle_path(all_paths, ref_sum, flag_sum,nsample)
    stat = {**stat, **buck_ratio}
    for k, v in dict_of_var_paths.items():
        path_sample_text = [x.text for x in v]
        extrinsic_eval_sample = eval_main(path_sample_text, ref_sum,flag_sum,k)
        stat = {**stat, **extrinsic_eval_sample}
    random_walk_dict = get_random_walk_eval(sos_key,all_nodes,all_edges)
    stat = {**stat, **random_walk_dict}
    
    logger.info(stat)
    return stat, net,dict_of_var_paths

from pathlib import Path

def analyze_main(config_name, dict_io_data, dict_io_text, dict_io_stat, dict_io_html):
    raw_files = os.listdir(os.path.join(dict_io_data, config_name))
    Path(os.path.join(dict_io_text, config_name)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(dict_io_stat, config_name)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(dict_io_html, config_name)).mkdir(parents=True, exist_ok=True)
    l = len(raw_files)
    # analyze_data(raw_files[0], config_name, dict_io_data=dict_io_data, dict_io_text=dict_io_text, dict_io_html=dict_io_html,dict_io_stat=dict_io_stat)
    for f in raw_files:
        analyze_data(f, config_name, dict_io_data=dict_io_data,
                 dict_io_text=dict_io_text, dict_io_html=dict_io_html,dict_io_stat=dict_io_stat)
    # with Pool(3) as pool:
    #     pool.starmap(analyze_data, zip(raw_files, [config_name]*l, [dict_io_data]*l, [dict_io_text]*l,  [dict_io_html]*l, [dict_io_stat]*l))


def analyze_data(f, config_name: str, dict_io_data: str, dict_io_text, dict_io_html , dict_io_stat):
    name = ".".join(f.split('.')[:-1])
    fname = os.path.join(dict_io_stat, config_name, f"{name}.json")
    if os.path.isfile(fname):
        logging.info(f"File exist")
        return 
    # summarization or translation?
    if config_name.startswith('sum'):
        flag_sum = True
    elif config_name.startswith('mt'):
        flag_sum = False
    else:
        raise ValueError("Task either sum or mt")

    with open(os.path.join(dict_io_data,config_name, f), 'rb') as fd:
        finished: SearchModelOutput = pickle.load(fd)
    logger.info(f)
    logger.info(finished.ends)
    if not finished.ends:
        return

    stat, network ,dict_of_var_paths = viz_result(finished.ends, finished.reference, flag_sum)

    # store text we only need str, rouge/bleu, 
    rt_json = {'src':finished.document,
                'tgt':finished.reference,
                'uid':finished.doc_id
    }
    for k,v in dict_of_var_paths.items():
        oracle_sents = [ (x.text, len(x), x.metrics) for x in v]
        rt_json[k] = oracle_sents
    with open(os.path.join(dict_io_text,config_name,  f"{name}.json"), 'w') as text_fd:
        json.dump(rt_json, text_fd)
    sample_as_pure_text = [x.text for x in dict_of_var_paths['sample']]
    with open(os.path.join(dict_io_text,config_name,  f"{name}.txt"), 'w') as text_fd:
        text_fd.write("\n".join(sample_as_pure_text))
    
    stat["file"] = name
    with open(fname, 'w') as wfd:
        json.dump(stat, wfd)
    network.save_graph(os.path.join(dict_io_html,config_name, f"{name}.html"))
    """
    with FileLock(f"{fname}.lock"):
        print("Lock acquired.")
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
    """


from src.recom_search.model.util import render_config_name
from src.recom_search.command.run_eval import run_model

from multiprocessing import Pool
from src.recom_search.model.setup import tokenizer, model, dataset, dec_prefix, args, dict_io
import logging
if __name__ == "__main__":
    # execute only if run as a script
    logging.info(f"Start running the pipeline")
    param_sim_function = {
        'ngram_suffix': args.ngram_suffix,
        'len_diff': args.len_diff,
        'merge': args.merge
    }
    config_search = {
        'post': args.post,
        'post_ratio': args.post_ratio,  # ratio of model calls left for post finishing
        'adhoc': args.adhoc,
        'heu': args.use_heu
    }
    combined_dict = {**config_search, **param_sim_function}
    combined_dict['avgsco'] = args.avg_score
    combined_dict['lenrwd'] = args.heu_seq_score_len_rwd
    combined_dict['topp'] = args.top_p
    config_name = render_config_name(
        args.task, args.dataset, args.model, args.beam_size, args.max_len, combined_dict)
    logging.info(f"Config name: {config_name}")
    analyze_main(config_name, dict_io['data'], dict_io['text'], dict_io['stat'], dict_io['html'])
