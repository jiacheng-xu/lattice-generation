"""

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
from src.recom_search.model.beam_node_ez import BeamNodeEz
from src.recom_search.model.heuristic import DeployHeu
from src.recom_search.model.util import pnum, render_name, run_inference_step

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

def search_frontier_committee(node, max_len):
    if node.finished:
        return False
    
def step_bfs(tokenizer, start_seed: BeamNodeEz, heap, doc_input_ids, model, use_heuristic: bool, avg_score, max_len: int, expl_steps: int, k_best: int, heu_func: DeployHeu) -> Tuple[Any, int]:
    step = 0
    finished_hypos = []
    pointer = start_seed
    cur_len = pointer.length
    if not expl_steps:
        expl_steps = max(1, max_len - cur_len)

    while (not pointer.finished) and step < expl_steps:
        step += 1
        dec_prefix = pointer.get_token_idx_as_input()
        _, output_prob, _, _ = run_inference_step(
            model, doc_input_ids, decoder_input_ids=dec_prefix, device=doc_input_ids.device, output_dec_hid=False)

        values, indices = torch.topk(output_prob, k=k_best)
        values = values[0].tolist()
        indices = indices[0].tolist()
        token_txt = tokenizer.decode(indices[0]).strip().lower()    # token of the decoded word

        top1_state = BeamNodeEz(prob=values[0], token_idx=indices[0], prev=[pointer], prev_score=[math.log(values[0])])

        seen_tokens = []
        if top1_state.finished or (step < expl_steps and (not top1_state.finished)):
            # if top1 has finished, we don't need to put it in frontier;
            # if we are still going to loop and it hasn't finished, we don't need to put it in frontier
            values = values[1:]
            indices = indices[1:]
            seen_tokens.append(token_txt)

        # if it hasn't finished and we are not going to loop, we need to put it in heap

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
            tmp_state = BeamNodeEz(prob=v, token_idx=i, prev=[pointer], prev_score=[math.log(v)])
            if tmp_state.finished or tmp_state.length >= max_len:  # if this branch is a completed one, just put it in the outputs.
                finished_hypos.append(tmp_state)
                continue
            
            """Start of some experimental code you can ignore"""
            if use_heuristic:
                heu_score = heu_func.run(cur_node=tmp_state, prev_len=cur_len, prob_distb=output_prob)
            else:
                heu_score = 0
            if avg_score > 0:
                model_score = tmp_state.get_score_sum() / tmp_state.length ** avg_score
            else:
                model_score = tmp_state.get_score_sum()
            """End"""

            score = model_score + heu_score
            """
            if random.random() < 0.01:
                logging.info(f"Score: {pnum(score)}\tModel: {pnum(model_score)}\tHeu: {pnum(heu_score)}")
            """
            heapq.heappush(heap, (-score, tmp_state))
        
        pointer = top1_state
        cur_len += 1
    return finished_hypos, step


def bfs(model, tokenizer,
           doc_input_ids: torch.LongTensor,
           config_heu: Optional[Dict],
           config_search: Optional[Dict],
           dec_prefix: Optional[List],
           avg_score,
           max_len: Optional[int],
           k_best: Optional[int],
           comp_budget: Optional[int]):

    ncalls = 0
    heu_func = DeployHeu(config_heu)
    
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
            init_seed = BeamNodeEz( prob=1., token_idx=prefix,
                                 prev=[last], prev_score=[0])
        else:
            init_seed = BeamNodeEz( prob=1., token_idx=prefix,
                                 prev=[], prev_score=[])
            last = init_seed

    heapq.heappush(heap, (-init_seed.prob, init_seed))

    while ncalls < budget_expl:
        _, seed = heapq.heappop(heap)   # top node from pq

        if config_search['dfs_expand']:
            expl_steps = max_len    # greedy depth-first expansion
        else:
            expl_steps = 1          # vanilla best first search only takes 1 step every time

        completed_hyps, added_num_calls = step_bfs(tokenizer, seed, heap, doc_input_ids, model,  config_search['heu'], avg_score, max_len=max_len, k_best=k_best, heu_func=heu_func, expl_steps=expl_steps)
        ncalls += added_num_calls  
        if completed_hyps:
            finished_hypos += completed_hyps
    num_comp_hypo_1_stage = len(finished_hypos)

    # If we still have budget post the vanilla bfs, we can do depth first expansion.
    while ncalls < comp_budget:
        _, seed = heapq.heappop(heap)
        expl_steps = max(1, max_len - seed.length)
        empty_heap = [] # at this stage, we do not put new nodes in the heap, hence we only need an empty heap. 
        completed_hyps, added_num_calls = step_bfs(tokenizer, seed, empty_heap, doc_input_ids, model,  config_search['heu'], avg_score, max_len=max_len, k_best=k_best, heu_func=heu_func, expl_steps=expl_steps)
        ncalls += added_num_calls
        if completed_hyps:
            finished_hypos += completed_hyps

    for hypo in finished_hypos:
        if not hypo.finished:
            logging.info(f"Not finished: {hypo}")
            continue
        logging.info(f"\n\n {hypo}")
        # hypo.print_lattice()
    logging.info(f"Fist Stage: {num_comp_hypo_1_stage}\Second Stage: {len(finished_hypos)}")
    return finished_hypos
