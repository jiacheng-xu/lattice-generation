"""

"""

import random
import torch
import math

from typing import Dict, List, Optional
import logging
import heapq

from src.recom_search.model.beam_node_full import BeamNodeFull
from src.recom_search.model.merge_strategy import merge_zip, merge_imp
from src.recom_search.model.bfs_util import  HashObject
from src.recom_search.model.heuristic import DeployHeu
from src.recom_search.model.merge import  similarity_heuristic
from src.recom_search.model.util import pnum, render_name, run_inference_step

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


def step_bfs_rcb_any(tokenizer, start_seed: BeamNodeFull, hash: HashObject, heap, doc_input_ids, model, param_sim_function, use_heuristic: bool, avg_score, max_len: int, expl_steps: int, k_best: int, heu_func: DeployHeu=None) -> Tuple[Any, int]:
    finished_hypos = []
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

        cur_dec_input_ids = pointer.all_token_idx
        dec_prefix = pointer.get_token_idx_as_input()
        _, output_prob, _, _ = run_inference_step(
            model, doc_input_ids, decoder_input_ids=dec_prefix, device=doc_input_ids.device, output_dec_hid=False)

        values, indices = torch.topk(output_prob, k=k_best)
        values = values[0].tolist()
        indices = indices[0].tolist()
        token_txt = tokenizer.decode(indices[0]).strip().lower()

        top1_state = BeamNodeFull(hash=hash,
                              prob=values[0], token_idx=indices[0], prev=[pointer.uid], prev_score=[math.log(values[0])])

        if cur_len >= hash.ngram:
            retrieved = hash.query(cur_dec_input_ids + [top1_state.token_idx])
            ngram = (cur_dec_input_ids +
                     [top1_state.token_idx])[-ngram_suffix:]

            for cand_par in retrieved:
                if cand_par == top1_state:
                    print("WHY?")
                    continue
                try:
                    one_match_path_token_ids, cand_par_suffix_node_ids = cand_par.get_tokens_match_suffix(ngram)
                except ValueError:
                    logging.error("Value error in finding suffix matching. ")
                    logging.error(cand_par, ngram)
                    continue
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

        seen_tokens = []
        if flag_merge or top1_state.finished or (step < expl_steps and (not top1_state.finished)):
            values = values[1:]
            indices = indices[1:]
            seen_tokens = [token_txt]

        if top1_state.finished or top1_state.length >= max_len:
            finished_hypos.append(top1_state)


        # add future candidate to heap
        for v, i in zip(values, indices):
            # remove cases like _XX, and XX if one of them already exist
            tok_txt = tokenizer.decode(i).strip().lower()
            if tok_txt in seen_tokens:
                continue
            else:
                seen_tokens.append(tok_txt)
            tmp_state = BeamNodeFull(hash=hash, prob=v, token_idx=i, prev=[pointer.uid], prev_score=[math.log(v)],master_node_uid=top1_state.uid)
            if tmp_state.finished or tmp_state.length >= max_len:  # if this branch is a completed one, just put it in the outputs.
                finished_hypos.append(tmp_state)
                continue
            if use_heuristic:
                heu_score = heu_func.run(cur_node=tmp_state, prev_len=cur_len, prob_distb=output_prob)
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

            heapq.heappush(heap, (-score, tmp_state.uid, pointer.uid))
        
        pointer = top1_state
        if pointer.finished or flag_merge:
            break
        cur_len += 1
    return finished_hypos, step
    # if flag_merge:
    #     return None, step
    # else:
    #     return finished_hypos, step

def bfs_rcb_any(model, tokenizer,
           doc_input_ids: torch.LongTensor,
           param_sim_function: Optional[Dict],
           config_heu: Optional[Dict],
           config_search: Optional[Dict],
           dec_prefix: Optional[List],
           avg_score:bool,
           max_len: Optional[int],
           k_best: Optional[int],
           comp_budget: Optional[int]):
    r"""

    """

    ncalls = 0
    heu_func = DeployHeu(config_heu)
    core_hash_obj = HashObject(param_sim_function['ngram_suffix'])
    heap = []  # nodes at the frontier of search
    finished_hypos = []
    # config_search.in: each time we expand a node, we always extend to end
    # config_search.post: after exploration, we try to extend all of the non-finished nodes until reach the budget
    assert not (config_search['post'] and config_search['dfs_expand'])
    if config_search['post']:
        budget_expl = comp_budget - \
            int(config_search['post_ratio'] * comp_budget)
    else:
        budget_expl = comp_budget

    last = None
    for prefix in dec_prefix:
        if last:
            init_seed = BeamNodeFull(hash=core_hash_obj, prob=1., token_idx=prefix,
                                 prev=[last.uid], prev_score=[0])
        else:
            init_seed = BeamNodeFull(hash=core_hash_obj, prob=1., token_idx=prefix,
                                 prev=[], prev_score=[])

            last = init_seed


    heapq.heappush(heap, (-init_seed.prob, init_seed.uid, last.uid))

    while ncalls < budget_expl:
        _, seed_uid, seed_prev_uid = heapq.heappop(heap)
        seed = core_hash_obj.retrieve_node(seed_uid)

        master_uid = seed.master_node_uid
        if master_uid != core_hash_obj.find_root_node_uid(master_uid):
            logging.info("Skip unexpanded children.")
            continue
        if seed_prev_uid != core_hash_obj.find_root_node_uid(seed_prev_uid):
            logging.info("Skip unexpanded children.")
            continue
        
        if config_search['dfs_expand']:
            expl_steps = max_len
        else:
            expl_steps = 1
        completed_hyps, added_num_calls = step_bfs_rcb_any(tokenizer,  seed, core_hash_obj, heap, doc_input_ids, model, param_sim_function, config_search['heu'], avg_score, max_len=max_len, k_best=k_best, heu_func=heu_func, expl_steps=expl_steps)

        ncalls += added_num_calls
        finished_hypos += completed_hyps

    num_comp_hypo_1_stage = len(finished_hypos)

    while ncalls < comp_budget:
        _, seed_uid, seed_prev_uid = heapq.heappop(heap)
        seed = core_hash_obj.retrieve_node(seed_uid)
        master_uid = seed.master_node_uid
        if master_uid != core_hash_obj.find_root_node_uid(master_uid):
            logging.info("Skip unexpanded children.")
            continue
        if seed_prev_uid != core_hash_obj.find_root_node_uid(seed_prev_uid):
            logging.info("Skip unexpanded children.")
            continue
        expl_steps = max(1, max_len - seed.length)
        empty_heap = [] # at this stage, we do not put new nodes in the heap, hence we only need an empty heap. 

        completed_hyps, added_num_calls = step_bfs_rcb_any(tokenizer,  seed, core_hash_obj, empty_heap, doc_input_ids, model, param_sim_function, config_search['heu'], avg_score, max_len=max_len, k_best=k_best, heu_func=heu_func, expl_steps=expl_steps)
        ncalls += added_num_calls
        finished_hypos += completed_hyps
    for hypo in finished_hypos:
        if not hypo.finished:
            logging.info(f"Not finished: {hypo}")
            continue
        logging.info(f"\n\n {hypo}")
        # hypo.print_lattice()
    logging.info(f"Fist Stage: {num_comp_hypo_1_stage}\Second Stage: {len(finished_hypos)}")
    return finished_hypos
