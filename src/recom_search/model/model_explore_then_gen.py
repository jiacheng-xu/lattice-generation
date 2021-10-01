
import random
import torch
from collections import defaultdict
import math
import pickle
from typing import List
import logging
from src.recom_search.model.model_bfs import explore_step
from src.recom_search.model.bfs_util import HashedGen
from src.recom_search.model.heuristic import DeployHeu
from src.recom_search.model.merge import core_merge, similarity_heuristic
from src.recom_search.model.util import pnum, render_name, run_inference_step
from src.recom_search.model.beam_state import BeamNode

import heapq


# first run vanilla best first search
# then generate and wrap up



def explore_then_gen(doc_input_ids, model, param_sim_function, eos_token_id=21, explore_steps=10, max_len=20, k_best=5, num_return_hypo=100, heu_config={}, debug: bool = False):
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

    while h and explored_cnt < 100:
        s = heapq.heappop(h)
        explored_cnt += 1
        prob, seed = s
        explore_step(seed,h,gen_hash,doc_input_ids,model,k_best, heu_func)


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

def exp_gen_generate():
    
    pass