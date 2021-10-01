
import random
import torch
from collections import defaultdict
import math
import pickle
from typing import List
import logging
from src.recom_search.model.heuristic import DeployHeu

from src.recom_search.model.merge import core_merge, similarity_heuristic
from src.recom_search.model.util import pnum, render_name, run_inference_step
from src.recom_search.model.beam_state import BeamNode

import heapq

from src.recom_search.model.bfs_util import HashedGen

def explore_step(start_seed:BeamNode, heap, hash_set:HashedGen, doc_input_ids, model, k_best, heu_func):
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



def generate_merge(start_seed, hash: HashedGen, eos_token_id, heap,  doc_input_ids, model, param_sim_function, max_len, explore_steps, k_best, heu_func: DeployHeu):
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
            tmp_state = BeamNode(prob=v, token_idx=i, prev=[
                                 pointer], prev_score=[math.log(v)])
            heu_score = heu_func.run(
                cur_node=tmp_state, prev_len=cur_len, prob_distb=output_prob)
            model_score = math.log(v)
            score = model_score + heu_score
            if random.random() < 0.01:
                logging.info(
                    f"Score: {pnum(score)}\tModel: {pnum(model_score)}\tHeu: {pnum(heu_score)}")
            heapq.heappush(heap, (-score, tmp_state))
        pointer = top1_state
        if pointer.finished or flag_merge:
            break
        cur_len += 1
    if flag_merge:
        return None, ncall
    else:
        return pointer, ncall


def best_first_search(doc_input_ids, model, param_sim_function, eos_token_id=21, explore_steps=10, max_len=20, k_best=5, num_return_hypo=100, heu_config={}, debug: bool = False):
    total_calls = 0
    explored_cnt = 0
    heu_func = DeployHeu(heu_config)
    hypos = []
    init_seed = BeamNode(prob=1., token_idx=eos_token_id,
                         prev=[], prev_score=[0])
    gen_hash = HashedGen(param_sim_function['ngram_suffix'])
    h = []
    heapq.heappush(h, (-init_seed.prob, init_seed))
    while h:
        s = heapq.heappop(h)
        explored_cnt += 1
        prob, seed = s
        output, ncall = generate_merge(start_seed=seed, hash=gen_hash, eos_token_id=eos_token_id, heap=h, doc_input_ids=doc_input_ids, model=model,
                                       param_sim_function=param_sim_function, max_len=max_len, explore_steps=explore_steps, k_best=k_best, heu_func=heu_func)
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

    fname = render_name(doc_input_ids, num_return_hypo, max_len,
                        param_sim_function, heu_config) + '.pkl'
    with open(f"vizs/best_{fname}", 'wb') as fd:
        pickle.dump(hypos, fd)
    return hypos
