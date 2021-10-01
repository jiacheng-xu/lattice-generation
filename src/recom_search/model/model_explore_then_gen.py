
import random
import torch
from collections import defaultdict
import math
import pickle
from typing import List
import logging

from src.recom_search.model.bfs_util import HashedGen
from src.recom_search.model.heuristic import DeployHeu
from src.recom_search.model.merge import core_merge, similarity_heuristic
from src.recom_search.model.util import pnum, render_name, run_inference_step
from src.recom_search.model.beam_state import BeamNode

import heapq


# first run vanilla best first search
# then generate and wrap up

import statistics


def gen_step(start_seed, hash: HashedGen,  doc_input_ids, model, param_sim_function, max_len, explore_steps=0):
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
        values, indices = torch.topk(output_prob, k=1)
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

        pointer = top1_state
        if pointer.finished or flag_merge:
            break
        cur_len += 1
    if flag_merge:
        return None, ncall
    else:
        return pointer, ncall


def explore_step(start_seed: BeamNode, heap, hash_set: HashedGen, doc_input_ids, model, k_best, heu_func):
    pointer = start_seed
    ncall = 0
    if pointer.finished:
        return
    if pointer.prev:
        hash_set.add_helper(pointer.prev[0], pointer)
    
    dec_prefix = pointer.get_token_idx_as_input()
    _, output_prob, _, _ = run_inference_step(
        model, doc_input_ids, decoder_input_ids=dec_prefix, device=doc_input_ids.device, output_dec_hid=False)
    ncall += 1
    values, indices = torch.topk(output_prob, k=k_best)
    values = values[0].tolist()
    indices = indices[0].tolist()

    for v, i in zip(values, indices):
        tmp_state = BeamNode(prob=v, token_idx=i, prev=[
            pointer], prev_score=[math.log(v)])
        heu_score = heu_func.run(
            cur_node=tmp_state, prev_len=pointer.length, prob_distb=output_prob)
        model_score = tmp_state.get_score_sum()
        score = model_score + heu_score
        if random.random() < 0.01:
            logging.info(
                f"Score: {pnum(score)}\tModel: {pnum(model_score)}\tHeu: {pnum(heu_score)}")
        heapq.heappush(heap, (-score, tmp_state))


def explore_then_gen(doc_input_ids, model, param_sim_function, eos_token_id=21, max_len=20, k_best=5, num_return_hypo=100, heu_config={}, debug: bool = False):
    total_calls = 0
    explored_cnt = 0
    heu_func = DeployHeu(heu_config)
    hypos = []
    init_seed = BeamNode(prob=1., token_idx=eos_token_id,
                         prev=[], prev_score=[0])
    gen_hash = HashedGen(param_sim_function['ngram_suffix'])
    h = []

    heapq.heappush(h, (-init_seed.prob, init_seed))

    # explore

    while h and explored_cnt < 30:
        s = heapq.heappop(h)
        explored_cnt += 1
        prob, seed = s

        explore_step(seed, h, gen_hash, doc_input_ids, model, k_best, heu_func)
        total_calls += 1

    # take a step back
    fathers = [x[1].prev[0] for x in h]
    fathers = list(set(fathers))
    avg_len_fathers = [f.length for f in fathers]
    logging.info(
        f"Num father: {len(fathers)}\tAvg len in frontier after explore: {statistics.quantiles(avg_len_fathers)}")

    hypos, gen_ncall = exp_gen_generate(
        h, gen_hash, doc_input_ids, model, param_sim_function, max_len)
    total_calls += gen_ncall
    logging.info(f"Number of calls: {total_calls}\n\n\n\n\n")
    for hypo in hypos:
        if not hypo.finished:
            logging.info(f"Not finished: {hypo}")
            continue
        logging.info(f"\n\n {hypo}")
        hypo.print_lattice()
    hypos = [x for x in hypos if x.finished]

    fname = render_name(doc_input_ids, num_return_hypo, max_len,
                        param_sim_function, heu_config) + '.pkl'
    with open(f"vizs/exp_gen_{fname}", 'wb') as fd:
        pickle.dump(hypos, fd)
    return hypos


def exp_gen_generate(heap, gen_hash, doc_input_ids, model, param_sim_function, max_len):
    total_calls = 0
    outputs = []
    while heap:
        seed = heap.pop()
        _, pointer = seed
        output, ncall = gen_step(
            pointer, gen_hash,  doc_input_ids, model, param_sim_function, max_len)
        total_calls += ncall
        if output:
            outputs.append(output)
    return outputs, total_calls
