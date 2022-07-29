from torch.distributions.categorical import Categorical
import pickle
from transformers.generation_logits_process import TopPLogitsWarper



from src.recom_search.model.beam_node import BeamNode
from src.recom_search.model.beam_node_ez import BeamNodeEz

from src.recom_search.model.merge import naive_merge, similarity_heuristic
from src.recom_search.model.util import run_inference_step, render_name
from typing import List
import logging
import torch
import math


def baseline_iterative_recomb(candidates: List[BeamNode], param_sim_function, beam_size):
    next_candidate: List[BeamNode] = []

    len_diff = param_sim_function['len_diff']
    ngram_suffix = param_sim_function['ngram_suffix']
    for candidate in candidates:
        flag_merge = False
        for nx_cand in next_candidate:
            # len of nx_cand
            # only check if the len diff of nx_cand and candidate is small
            pointer = nx_cand

            len_cand = candidate.length
            while abs(len_cand - pointer.length) < len_diff:
                flag = similarity_heuristic(
                    candidate.all_token_idx, pointer.all_token_idx, ngram_suffix, len_diff)
                if not flag:
                    if pointer.prev:
                        pointer = pointer.prev[0]
                        continue
                    else:
                        break
                if pointer.get_score_sum() > candidate.get_score_sum():
                    # merge happens
                    flag_merge = True
                    break
                else:
                    # candidate get better score?
                    break
            if flag_merge:
                break
        if flag_merge:
            # core_merge(pointer, candidate)
            naive_merge(pointer, candidate)
        else:
            next_candidate.append(candidate)
        if len(next_candidate) >= beam_size:
            return next_candidate
    return next_candidate

def gen_init_seed_with_dec_prefix(dec_prefix)->BeamNodeEz:
    last = None
    for prefix in dec_prefix:
        if last:
            init_seed = BeamNodeEz( prob=1., token_idx=prefix,
                                 prev=[last], prev_score=[0])
        else:
            init_seed = BeamNodeEz( prob=1., token_idx=prefix,
                                 prev=[], prev_score=[])

        last = init_seed
    return last

def baseline_recomb_sample(doc_input_ids, dec_prefix, model, param_sim_function, max_len=20, num_return_hypo=100,  top_p=0.8):
    topp_logit_wrapper = TopPLogitsWarper(top_p=top_p)
    """Neucleus sampling with path recombination"""
    # budget = max_len * beam size
    total_budget = max_len * num_return_hypo
    usage = 0

    len_diff = param_sim_function['len_diff']
    ngram_suffix = param_sim_function['ngram_suffix']

    gen_nodes = {}

    init_seed = gen_init_seed_with_dec_prefix(dec_prefix)
    hypo = init_seed
    
    ends = []
    while True:
        # sample start from a sentence, do not create a new node if something matches the record
        # create new node, try to recomb,
        # stop when the budget runs out

        decoder_input_ids = hypo.get_token_idx_as_input()
        output_tokens, output_prob, output_score, _ = run_inference_step(
            model, doc_input_ids, decoder_input_ids=decoder_input_ids, device=doc_input_ids.device, output_dec_hid=False, T=1)
        usage += 1
        sample_output = topp_logit_wrapper(None, scores=output_score).squeeze()
        
        m = Categorical(logits=sample_output)
        
        next_tok_idx = m.sample().cpu().item()
        next_tok_prob = output_prob[0][next_tok_idx]
        # has this token combination been pred before?
        feat = hypo.all_token_idx + [next_tok_idx]
        feat = [str(x) for x in feat]
        feat_key = "_".join(feat)

        if feat_key in gen_nodes:
            tmp_node = gen_nodes[feat_key]
            # the same path has been generated before, so we do not need to recombine
            merge_flag = False
        else:
            merge_flag = True
            tmp_node = BeamNodeEz(prob=next_tok_prob, token_idx=next_tok_idx, prev=[
                                hypo], prev_score=[math.log(next_tok_prob)])
            gen_nodes[feat_key] = tmp_node

        # try to recombine
        merge_happen = False
        if merge_flag:

            tmp_node_token_idx = tmp_node.all_token_idx
            for node_key, node in gen_nodes.items():
                if node_key == feat_key:
                    continue
                flag = similarity_heuristic(
                    tmp_node_token_idx, node.all_token_idx, ngram_suffix, len_diff)
                if flag:
                    naive_merge(node, tmp_node)
                    merge_happen = True
                    break
        # if not finished, move on, else, reset
        if merge_happen or tmp_node.finished or tmp_node.length > max_len:
            hypo = init_seed
            merge_flag = False
            # if tmp_node.finished and not merge_happen:
            if not merge_happen:   # max_len truncated
                ends.append(tmp_node)
        else:
            hypo = tmp_node
        if total_budget <= usage:
            break

    return ends


def recomb_baseline(doc_input_ids, dec_prefix, model, param_sim_function, beam_size=5, max_len=20, avg_score:float=-1, debug: bool = False):
    init_seed = gen_init_seed_with_dec_prefix(dec_prefix)

    hypos:List[BeamNodeEz] = [init_seed]
    finished = []
    for t in range(max_len):
        # TODO finished
        candidates:List[BeamNodeEz] = []
        for hypo in hypos:
            
            if hypo.finished:
                finished.append(hypo)
                continue
            # prefix
            decoder_input_ids = hypo.get_token_idx_as_input()
            output_tokens, output_prob, output_score, _ = run_inference_step(model, doc_input_ids, decoder_input_ids=decoder_input_ids, device=doc_input_ids.device, output_dec_hid=False, T=1)

            values, indices = torch.topk(output_prob, k=beam_size)
            values = values[0].tolist()
            indices = indices[0].tolist()

            for idx, v, i in zip(range(beam_size), values, indices):
                tmp_state = BeamNodeEz(prob=v, token_idx=i, prev=[hypo], prev_score=[math.log(v)])
                candidates.append(tmp_state)

        # sort candidates by scores; these are active candidates of the current step
        if avg_score == -1:
            sorted_candidates = sorted(
            candidates, key=lambda x: x.get_score_sum(), reverse=True)
        else:
            sorted_candidates = sorted(
            candidates, key=lambda x: x.get_score_sum() / len(x) ** avg_score, reverse=True)
        hypos = baseline_iterative_recomb(
            sorted_candidates, param_sim_function, beam_size=beam_size)
        logging.info('-'*30)
    # sort finished
    if avg_score == -1:
        finished = sorted(
        finished, key=lambda x: x.get_score_sum(), reverse=True)
    else:
        finished = sorted(
        finished, key=lambda x: x.get_score_sum() / len(x) ** avg_score, reverse=True)
    finished = finished[:beam_size]
    return finished
