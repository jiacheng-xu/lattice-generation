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


from src.recom_search.model.bfs_util import HashedGen
from src.recom_search.model.heuristic import DeployHeu
from src.recom_search.model.merge import core_merge, similarity_heuristic
from src.recom_search.model.util import pnum, render_name, run_inference_step, setup_logger
from src.recom_search.model.beam_state import BeamNode
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


def astar_step(start_seed: BeamNode, hash: HashedGen, heap,  doc_input_ids, model, param_sim_function, use_heuristic: bool, max_len: int, expl_steps: int, k_best: int, heu_func: DeployHeu)->Tuple[Any,int]:
    cnt_call = 0
    step = 0
    ngram_suffix = param_sim_function['ngram_suffix']
    len_diff = param_sim_function['len_diff']
    pointer = start_seed
    cur_len = pointer.length
    if not expl_steps:
        expl_steps = max(1,max_len - cur_len)
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
        top1_state = BeamNode(
            prob=values[0], token_idx=indices[0], prev=[pointer], prev_score=[math.log(values[0])])
        """
        values = values[1:]
        indices = indices[1:]
        """

        # is top1 in hash?
        if cur_len >= hash.ngram:
            retrieved = hash.query(cur_dec_input_ids + [top1_state.token_idx])
            ngram = (cur_dec_input_ids +
                     [top1_state.token_idx])[-ngram_suffix:]

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
        
        
        if not flag_merge:
            hash.add_helper(pointer, top1_state)
        
        if flag_merge or top1_state.finished or (step < expl_steps and (not top1_state.finished)):
            values = values[1:]
            indices = indices[1:]

        # add future candidate to heap
        for v, i in zip(values, indices):
            tmp_state = BeamNode(prob=v, token_idx=i, prev=[
                pointer], prev_score=[math.log(v)])
            if use_heuristic:
                heu_score = heu_func.run(
                    cur_node=tmp_state, prev_len=cur_len, prob_distb=output_prob)
            else:
                heu_score = 0
            model_score = tmp_state.get_score_sum()
            score = model_score + heu_score
            if random.random() < 0.01:
                logging.info(
                    f"Score: {pnum(score)}\tModel: {pnum(model_score)}\tHeu: {pnum(heu_score)}")
            # print(score)
            heapq.heappush(heap, (-score, tmp_state))
        pointer = top1_state
        if pointer.finished or flag_merge:
            break
        cur_len += 1
    if flag_merge:
        return None, cnt_call
    else:
        return pointer, cnt_call


def a_star(model, doc_input_ids: torch.LongTensor,  param_sim_function: Optional[Dict], config_heu: Optional[Dict], config_search: Optional[Dict],  eos_token_id: Optional[int], max_len: Optional[int], k_best: Optional[int], comp_budget: Optional[int]):
    r"""

    """

    ncalls = 0
    heu_func = DeployHeu(config_heu)
    gen_hash = HashedGen(param_sim_function['ngram_suffix'])
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
    init_seed = BeamNode(prob=1., token_idx=eos_token_id,
                         prev=[], prev_score=[])
    heapq.heappush(heap, (-init_seed.prob, init_seed))

    while ncalls < budget_expl:
        _, seed  = heapq.heappop(heap)
        if config_search['adhoc']:
            expl_steps=max_len
        else:
            expl_steps = 1
        output_node, added_num_calls = astar_step(seed, gen_hash,heap,doc_input_ids,model,param_sim_function,config_search['heu'],max_len=max_len,k_best=k_best,heu_func=heu_func,expl_steps=expl_steps)

        ncalls += added_num_calls
        
        if output_node and output_node.finished:
            finished_hypos.append(output_node)
    num_mid_point_hypo = len(finished_hypos)
    # if there is post generation (like explore-then-gen)
    while ncalls < comp_budget:
        _, seed  = heapq.heappop(heap)
        expl_steps=max(1,max_len - seed.length)
        output_node, added_num_calls = astar_step(seed, gen_hash,[],doc_input_ids,model,param_sim_function,config_search['heu'],max_len=max_len,k_best=k_best,heu_func=heu_func,expl_steps=expl_steps)

        ncalls += added_num_calls
        
        if output_node and output_node.finished:
            finished_hypos.append(output_node)

    ### check what's in heap, hash
    # print(heap)
    # print(gen_hash)
    ###
    
    for hypo in finished_hypos:
        if not hypo.finished:
            logging.info(f"Not finished: {hypo}")
            continue
        logging.info(f"\n\n {hypo}")
        hypo.print_lattice()
    logging.info(f"Mid: {num_mid_point_hypo}\tEnd: {len(finished_hypos)}")
    return finished_hypos

def main():

    config_search = {
        'post': True,
        'post_ratio': 0.7,  # ratio of model calls left for post finishing
        'adhoc': False,
        'heu': False
    }
    config_search = {
        'post': False,
        'post_ratio': 0.7,  # ratio of model calls left for post finishing
        'adhoc': False,
        'heu': True
    }
    param_sim_function = {
        'ngram_suffix': 3,
        'len_diff': 5
    }
    config_heu = {}
    input_doc = "Southwest Airlines and American Airlines, both based in Texas, said Tuesday that they will continue plans to require employees to get vaccinated, despite an edict issued by Texas Gov. Greg Abbott that would ban vaccine mandates for private businesses in the state."
    doc_input_ids = tokenizer(input_doc, return_tensors="pt").input_ids.to(device)
    a_star(model=model, doc_input_ids=doc_input_ids, k_best=5, comp_budget=200, config_search=config_search,
           max_len=23, eos_token_id=tokenizer.eos_token_id, param_sim_function=param_sim_function, config_heu=config_heu)


if __name__ == "__main__":
    setup_logger('test')
    from src.recom_search.model.util import model, tokenizer,device
    main()
