"""
Score could be model score + optional heuristic score.
Expansion strategy:
vanilla
Optional greedy_end: during the search or after the search (explore-then-gen)

"""

import random
import torch
from collections import UserDict, defaultdict
import math
import pickle
from typing import Dict, List, Optional
import logging
import heapq
import statistics
from src.recom_search.model.new_merge import merge_zip, merge_imp


from src.recom_search.model.bfs_util import  NewHash
from src.recom_search.model.heuristic import DeployHeu
from src.recom_search.model.merge import new_core_merge, similarity_heuristic
from src.recom_search.model.util import pnum, render_name, run_inference_step
from src.recom_search.model.beam_state import BeamNode
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


def astar_step(tokenizer, force_dec_prefix, start_seed: BeamNode, hash: NewHash, heap,  doc_input_ids, model, param_sim_function, use_heuristic: bool, avg_score, max_len: int, expl_steps: int, k_best: int, heu_func: DeployHeu) -> Tuple[Any, int]:
    cnt_call = 0
    step = 0
    ngram_suffix = param_sim_function['ngram_suffix']
    len_diff = param_sim_function['len_diff']
    merge_method = param_sim_function['merge']
    pointer = start_seed
    cur_len = pointer.length
    if not expl_steps:
        expl_steps = max(1, max_len - cur_len)
    flag_merge = False

    while (not pointer.finished) and step < expl_steps:
        step += 1
        cnt_call += 1

        cur_dec_input_ids = pointer.all_token_idx
        dec_prefix = pointer.get_token_idx_as_input()
        _, output_prob, _, _ = run_inference_step(
            model, doc_input_ids, decoder_input_ids=dec_prefix, device=doc_input_ids.device, output_dec_hid=False)

        values, indices = torch.topk(output_prob, k=k_best)
        values = values[0].tolist()
        indices = indices[0].tolist()
        token_txt = tokenizer.decode(indices[0]).strip().lower()

        top1_state = BeamNode(hash=hash,
                              prob=values[0], token_idx=indices[0], prev=[pointer.uid], prev_score=[math.log(values[0])])
        # hash.set_node(top1_state.uid, top1_state)

        if cur_len >= hash.ngram:
            retrieved = hash.query(cur_dec_input_ids + [top1_state.token_idx])
            ngram = (cur_dec_input_ids +
                     [top1_state.token_idx])[-ngram_suffix:]
            # print('===========')
            # print(ngram,tokenizer.decode(ngram))
            # print(retrieved)
            for cand_par in retrieved:
                if cand_par == top1_state:
                    print("WHY?")
                    continue
                one_match_path_token_ids, cand_par_suffix_node_ids = cand_par.get_tokens_match_suffix(ngram)
                flag = similarity_heuristic(
                    one_match_path_token_ids, top1_state.all_token_idx, ngram_suffix, len_diff)
                if flag:
                    flag_merge = True
                    break
            if flag_merge:
                if merge_method == 'zip':
                    merge_zip(hash, cand_par, top1_state, par_match_uids=cand_par_suffix_node_ids)
                elif merge_method == 'imp':
                    flag = merge_imp(hash, cand_par, top1_state)
                    if not flag:
                        flag_merge = False
                else:
                    raise NotImplementedError

        if not flag_merge:
            hash.add_helper(pointer, top1_state)

        if flag_merge or top1_state.finished or (step < expl_steps and (not top1_state.finished)):
            values = values[1:]
            indices = indices[1:]
            seen_tokens = [token_txt]
        else:
            seen_tokens = []
            # print()
            # when can this happen?
            pass

        # add future candidate to heap
        for v, i in zip(values, indices):

            # remove cases like _XX, and XX if one of them already exist
            tok_txt = tokenizer.decode(i).strip().lower()
            if tok_txt in seen_tokens:
                continue
            else:
                seen_tokens.append(tok_txt)
            tmp_state = BeamNode(hash=hash, prob=v, token_idx=i, prev=[pointer.uid], prev_score=[math.log(v)],master_node_uid=top1_state.uid)
            if use_heuristic:
                heu_score = heu_func.run(
                    cur_node=tmp_state, prev_len=cur_len, prob_distb=output_prob)
            else:
                heu_score = 0
            if avg_score > 0:
                model_score = tmp_state.get_score_sum() / tmp_state.length ** avg_score
            else:
                model_score = tmp_state.get_score_sum()
            score = model_score + heu_score
            if random.random() < 0.01:
                logging.info(
                    f"Score: {pnum(score)}\tModel: {pnum(model_score)}\tHeu: {pnum(heu_score)}")
            # print(score)
            heapq.heappush(heap, (-score, tmp_state.uid))
        
        pointer = top1_state
        if pointer.finished or flag_merge:
            break
        cur_len += 1
    if flag_merge:
        return None, cnt_call
    else:
        return pointer, cnt_call


def a_star(model, tokenizer,
           doc_input_ids: torch.LongTensor,
           param_sim_function: Optional[Dict],
           config_heu: Optional[Dict],
           config_search: Optional[Dict],
           dec_prefix: Optional[List],
           avg_score,
           max_len: Optional[int],
           k_best: Optional[int],
           comp_budget: Optional[int]):
    r"""

    """

    ncalls = 0
    heu_func = DeployHeu(config_heu)
    
    new_hash = NewHash(param_sim_function['ngram_suffix'])
    heap = []  # nodes at the frontier of search
    finished_hypos = []
    # config_search.in: each time we expand a node, we always extend to end
    # config_search.post: after exploration, we try to extend all of the non-finished nodes until reach the budget
    assert not (config_search['post'] and config_search['adhoc'])
    if config_search['post']:
        budget_expl = comp_budget - \
            int(config_search['post_ratio'] *
                comp_budget)
    else:
        budget_expl = comp_budget

    last = None
    for prefix in dec_prefix:
        if last:
            init_seed = BeamNode(hash=new_hash, prob=1., token_idx=prefix,
                                 prev=[last.uid], prev_score=[0])
        else:
            init_seed = BeamNode(hash=new_hash, prob=1., token_idx=prefix,
                                 prev=[], prev_score=[])

            last = init_seed
        # new_hash.set_node(init_seed.uid, init_seed)

    heapq.heappush(heap, (-init_seed.prob, init_seed.uid))

    while ncalls < budget_expl:
        _, seed_uid = heapq.heappop(heap)
        seed = new_hash.retrieve_node(seed_uid)

        master_uid = seed.master_node_uid
        if master_uid != new_hash.find_root_node_uid(master_uid):
            print("Skipping unexpanded children!")
            continue
        
        if config_search['adhoc']:
            expl_steps = max_len
        else:
            expl_steps = 1
        output_node, added_num_calls = astar_step(tokenizer, dec_prefix, seed, new_hash, heap, doc_input_ids, model, param_sim_function, config_search['heu'], avg_score, max_len=max_len, k_best=k_best, heu_func=heu_func, expl_steps=expl_steps)

        ncalls += added_num_calls
        print(output_node)
        if output_node and output_node.finished:
            finished_hypos.append(output_node)
    num_mid_point_hypo = len(finished_hypos)
    # if there is post generation (like explore-then-gen)
    while ncalls < comp_budget:
        _, seed_uid = heapq.heappop(heap)
        if seed_uid != new_hash.find_root_node_uid(seed_uid):
            print("Skipping unexpanded children!")
            continue
        # seed = new_hash.retrieve_node(seed_uid)
        _, seed = heapq.heappop(heap)
        expl_steps = max(1, max_len - seed.length)
        output_node, added_num_calls = astar_step(tokenizer, dec_prefix, seed, new_hash, [], doc_input_ids, model, param_sim_function,
                                                  config_search['heu'], avg_score, max_len=max_len, k_best=k_best, heu_func=heu_func, expl_steps=expl_steps)

        ncalls += added_num_calls

        if output_node and output_node.finished:
            finished_hypos.append(output_node)

    # check what's in heap, hash
    # print(heap)
    # print(gen_hash)
    ###

    for hypo in finished_hypos:
        if not hypo.finished:
            logging.info(f"Not finished: {hypo}")
            continue
        logging.info(f"\n\n {hypo}")
        # hypo.print_lattice()
    logging.info(f"Mid: {num_mid_point_hypo}\tEnd: {len(finished_hypos)}")
    return finished_hypos
