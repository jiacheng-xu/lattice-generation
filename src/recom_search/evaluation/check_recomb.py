from src.recom_search.model.setup import tokenizer, model, args
import os
from tqdm import tqdm
import torch
from collections import defaultdict
import enum
import statistics
import sys
from src.recom_search.evaluation.eval_bench import rouge_single_pair
from src.recom_search.model.beam_node import BeamNode
from src.recom_search.evaluation.analysis import find_start_end
from src.recom_search.model.setup import setup_model
import random
import pickle
nsample =1000


def func_one_file(data):
    generated_outputs = data.ends

    if len(generated_outputs) == 0:
        return {}

    # first set all nodes and edges
    all_nodes = {}
    all_edges = {}

    for idx, go in enumerate(generated_outputs):
        nodes, edges = go.visualization()
        hash_obj = go.hash
        all_nodes.update(nodes)
        all_edges.update(edges)
    sos_key, list_of_eos_key, degree_mat = find_start_end(nodes, edges)
    output = []

    seen = []

    def dfs(node_id):
        real_node_id = hash_obj.find_root_node_uid(node_id)
        if node_id in seen:
            return
        if real_node_id in seen:
            return

        if node_id not in seen:
            seen.append(node_id)
        if real_node_id not in seen:
            seen.append(real_node_id)

        node: BeamNode = hash_obj.retrieve_node(real_node_id)
        fathers = node.prev
        # dedup
        fathers = [hash_obj.find_root_node_uid(f) for f in fathers]
        fathers = list(set(fathers))
        cur_tok_id = node.token_idx
        if len(fathers) > 1:
            l = len(fathers)
            for idx, f in enumerate(fathers):
                prev_node = hash_obj.retrieve_node(f)
                prev_tok = prev_node.all_token_idx
                for jdx in range(idx+1, l):
                    another_f = fathers[jdx]
                    sec_prev_node = hash_obj.retrieve_node(another_f)
                    sec_prev_tok = sec_prev_node.all_token_idx
                    output.append((data.document, prev_tok +
                                  [cur_tok_id], sec_prev_tok+[cur_tok_id]))
        for f in fathers:
            dfs(f)
        # seen.append(node)
        return

    for node in list_of_eos_key:
        dfs(node)
    if not output:
        return []
    # print(list_of_eos_key)
    # print(output)
    trunc_output = random.choices(output, k=10)
    return trunc_output, len(output)


bucket = [1, 2, 4, 8, 16]


def run_check(all_recombs, folder_name, device):
    random.shuffle(all_recombs)
    all_recombs = all_recombs[:nsample]
    total_cnt = 0
    em = defaultdict(list)
    rouges_2 = defaultdict(list)

    for rec in tqdm(all_recombs):
        document, seq_a, seq_b = rec
        sents = document.split('\n')
        inp = "\n".join(sents[:10])[:5000]
        input_ids = tokenizer(inp, return_tensors="pt").input_ids
        len_a, len_b = len(seq_a), len(seq_b)
        input_ids = torch.LongTensor(input_ids).to(device)
        seq_a = torch.LongTensor(seq_a).to(device).unsqueeze(0)
        output_a = model.generate(
            input_ids=input_ids, decoder_input_ids=seq_a).cpu().tolist()[0]
        # print(output_a)
        seq_b = torch.LongTensor(seq_b).to(device).unsqueeze(0)
        output_b = model.generate(
            input_ids=input_ids, decoder_input_ids=seq_b).cpu().tolist()[0]
        total_cnt += 1
        a_suffix = output_a[len_a:]
        b_suffix = output_b[len_b:]
        # print(tokenizer.decode(output_a[:len_a], skip_special_tokens=True),
        #       "======", tokenizer.decode(a_suffix, skip_special_tokens=True))
        # print(tokenizer.decode(output_b[:len_b], skip_special_tokens=True),
        #       "======", tokenizer.decode(b_suffix, skip_special_tokens=True))
        # print('')
        for buck in bucket:
            tmp_a_suffix = a_suffix[:buck]
            tmp_b_suffix = b_suffix[:buck]
            if tmp_a_suffix == tmp_b_suffix:
                em[buck].append(1)
            else:
                em[buck].append(0)
            str_a, str_b = tokenizer.decode(tmp_a_suffix, skip_special_tokens=True), tokenizer.decode(
                tmp_b_suffix, skip_special_tokens=True)
            # rouge_1_f1 = rouge_single_pair(str_a, str_b)
            rouge_2_f1 = rouge_single_pair(str_a, str_b, 'rouge2')
            # rouges[buck].append(rouge_1_f1)
            rouges_2[buck].append(rouge_2_f1)
    lines = []
    for buck in bucket:
        r = rouges_2[buck]
        s = f"{buck}\t{statistics.mean(r)}\t{statistics.mean(em[buck])}"
        lines.append(s)
    with open(f"recomb_{folder_name}.txt", 'w') as fd:
        fd.write("\n".join(lines))


def main(directory, folder, device='cuda:0') -> int:
    """Echo the input arguments to standard output"""
    files = os.listdir(os.path.join(directory, folder))
    print(folder)
    # print(files)
    recombs = []
    mergs = []
    for f in files:
        with open(os.path.join(directory, folder, f), 'rb') as rfd:
            data = pickle.load(rfd)
        # find all nodes with >1 prev
        one_output, nmerges = func_one_file(data)
        recombs += one_output
        mergs.append(nmerges)
    run_check(recombs, folder, device=device)


if __name__ == '__main__':
    directory = '/mnt/data1/jcxu/lattice-sum/output/data/'
    folders = os.listdir(directory)
    folders = [f for f in folders if 'sum_xsum_astar' in f]
    for d in folders:
        print(d)
        main(directory, d, args.device) 
