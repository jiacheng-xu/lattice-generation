
import torch
from collections import defaultdict
import math
import pickle
from typing import List
import logging

from src.recom_search.model.merge import core_merge, similarity_heuristic
from src.recom_search.model.util import render_name, run_inference_step
from src.recom_search.model.beam_state import BeamNode

import heapq


class HashedGen():
    def __init__(self, ngram: int = 5) -> None:
        self.data = defaultdict(list)
        self.ngram = ngram

    def const_key(self, token_ids):
        tokens = token_ids[-self.ngram:]
        token_str = [str(x) for x in tokens]
        k = "_".join(token_str)
        return k

    def query(self, token_ids: List[int]):
        # get the last n tokens
        if len(token_ids) < self.ngram:
            return []
        k = self.const_key(token_ids)
        if k in self.data:
            return self.data[k]
        else:
            return []

    def add(self, node):
        tokens = node.all_token_idx
        if len(tokens) < self.ngram:
            return
        k = self.const_key(tokens)
        self.data[k].append(node)

    def add_helper(self, par_node, new_node):
        # par_node : the parent node

        def dfs(node: BeamNode, depth):
            if not node:
                return []
            if depth == self.ngram:
                return [[node.token_idx]]
            prevs = node.prev
            outputs = []
            for p in prevs:
                many = dfs(p, depth+1)
                for one in many:
                    outputs.append(one + [node.token_idx])
            return outputs
        all_probable_paths = dfs(par_node, 1)
        all_probable_paths = [x + [new_node.token_idx]
                              for x in all_probable_paths if len(x) == self.ngram]

        cnt = 0
        for p in all_probable_paths:
            key = self.const_key(p)
            self.data[key].append(new_node)
            cnt += 1
        # logging.debug(f"{cnt} added to Hash.")


def generate_merge(start_seed, hash: HashedGen, eos_token_id, heap,  doc_input_ids, model, param_sim_function, max_len, explore_steps, k_best, position_bias):
    # try to extend the start_seed for explore_steps steps. if there is a mergable match, do that match, else, finish the generation
    ncall = 0
    ngram_suffix = param_sim_function['ngram_suffix']
    len_diff = param_sim_function['len_diff']

    pointer = start_seed

    cur_len = pointer.length
    if explore_steps > 0:
        target_steps = min(cur_len + explore_steps, max_len)
    else:
        target_steps = max_len
    flag_merge = False
    while cur_len < max_len:
        if pointer.finished:
            break
        cur_dec_input_ids = pointer.all_token_idx
        dec_prefix = pointer.get_token_idx_as_input()
        _, output_prob, _, _ = run_inference_step(
            model, doc_input_ids, decoder_input_ids=dec_prefix, device=doc_input_ids.device, output_dec_hid=False)
        ncall += 1
        values, indices = torch.topk(output_prob, k=k_best)
        values = values[0].tolist()
        indices = indices[0].tolist()

        top1_state = BeamNode(
            prob=values[0], token_idx=indices[0], prev=[pointer], prev_score=[math.log(values[0])])
        values = values[1:]
        indices = indices[1:]
        # is top1 in hash?
        if cur_len < target_steps and cur_len >= hash.ngram:
            retrieved = hash.query(cur_dec_input_ids + [top1_state.token_idx])
            ngram = (cur_dec_input_ids +
                     [top1_state.token_idx])[-ngram_suffix:]
            if retrieved:   # are there possible hash there?
                for candidate_pair in retrieved:
                    span_end = candidate_pair
                    one_match_path_token_ids = span_end.get_tokens_match_suffix(
                        ngram)
                    # print(one_match_path_token_ids)
                    # print(top1_state.all_token_idx)
                    flag = similarity_heuristic(
                        one_match_path_token_ids, top1_state.all_token_idx, ngram_suffix, len_diff)
                    if flag:
                        flag_merge = True
                        break
                if flag_merge:
                    core_merge(span_end, top1_state)
                    

        # add stuff to heap
        if not flag_merge:
            hash.add_helper(pointer, top1_state)
        # else:
        #     print()
        for v, i in zip(values, indices):
            tmp_state = BeamNode(prob=v, token_idx=i, prev=[pointer],prev_score=[math.log(v)])
            if position_bias > 1:
                score = v - math.log((cur_len+1)/max_len)/position_bias
            else:
                score = v
            heapq.heappush(heap, (-score, tmp_state))
        pointer = top1_state
        if pointer.finished or flag_merge:
            break
        cur_len += 1
    if flag_merge:
        return None, ncall
    else:
        return pointer, ncall


def best_first_search(doc_input_ids, model, param_sim_function, eos_token_id=21, explore_steps=10, max_len=20, k_best=5, num_return_hypo=100, position_bias=0.0, debug: bool = False):
    total_calls = 0
    explored_cnt = 0
    hypos = []
    init_seed = BeamNode(prob=1., token_idx=eos_token_id, prev=[], prev_score=[0])
    gen_hash = HashedGen(param_sim_function['ngram_suffix'])
    h = []
    heapq.heappush(h, (-init_seed.prob, init_seed))
    while h:
        s = heapq.heappop(h)
        explored_cnt += 1
        prob, seed = s
        output, ncall = generate_merge(start_seed=seed, hash=gen_hash, eos_token_id=eos_token_id, heap=h, doc_input_ids=doc_input_ids, model=model,
                                       param_sim_function=param_sim_function, max_len=max_len, explore_steps=explore_steps, k_best=k_best, position_bias=position_bias)
        total_calls += ncall
        if output:
            hypos.append(output)
        if total_calls >= num_return_hypo:
            break

    logging.info(f"#Whole Beam: {len(hypos)} ")
    logging.info('\n\n\n\n\n')
    for hypo in hypos:
        if not hypo.finished:
            logging.info(f"Not finished: {hypo}")
            continue
        logging.info(f"\n\n {hypo}")
        hypo.print_lattice()
    hypos = [x for x in hypos if x.finished]
    outputs = []

    fname = render_name(doc_input_ids, num_return_hypo, max_len,
                        param_sim_function['ngram_suffix'], param_sim_function['len_diff'], position_bias) + '.pkl'
    with open(f"vizs/best_{fname}", 'wb') as fd:
        pickle.dump(hypos, fd)
    return outputs
