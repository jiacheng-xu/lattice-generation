
import pickle
from src.recom_search.model.beam_state import BeamNode

from src.recom_search.model.merge import core_merge, similarity_heuristic
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
            core_merge(pointer, candidate)
        else:
            next_candidate.append(candidate)
        if len(next_candidate) >= beam_size:
            return next_candidate
    return next_candidate


def recomb_baseline(doc_input_ids, model, param_sim_function, eos_token_id=21, beam_size=5, max_len=20, num_return_hypo=100, debug: bool = False):
    # gen_hash = GenHash(ngram=param_sim_function['ngram_suffix'])

    hypos = [BeamNode(prob=1.0, token_idx=eos_token_id, prev=[],prev_score=[])]

    for t in range(max_len):
        # TODO finished
        candidates = []
        for hypo in hypos:
            if not debug:
                if hypo.finished:
                    candidates.append(hypo)
                    continue
                # prefix
                decoder_input_ids = hypo.get_token_idx_as_input()
                output_tokens, output_prob, output_score, _ = run_inference_step(
                    model, doc_input_ids, decoder_input_ids=decoder_input_ids, device=doc_input_ids.device, output_dec_hid=False, T=1)

                # pred_entropy = entropy(output_prob.cpu().numpy(), axis=-1)[0]
                # print(pnum(pred_entropy))
                # dynamic_k = min(BS, t+1)

                values, indices = torch.topk(output_prob, k=beam_size)
                values = values[0].tolist()
                indices = indices[0].tolist()
                # trim
                # values = [x for x in values if x > 0.01]
                # indices = indices[:len(values)]
            else:
                # replace it with something real
                values, indices = model(t, beam_size)
                # values are list of probs sum<1, indices are token idx

            for idx, v, i in zip(range(beam_size), values, indices):

                tmp_state = BeamNode(prob=v, token_idx=i, prev=[hypo], prev_score=[math.log(v)])
                # gen_hash.add(beam_item.token_full + [indices[idx]],tmp_state)
                candidates.append(tmp_state)

        # sort candidates by scores; these are active candidates of the current step
        sorted_candidates = sorted(
            candidates, key=lambda x: x.get_score_avg(), reverse=True)
        hypos = baseline_iterative_recomb(
            sorted_candidates, param_sim_function, beam_size=beam_size)
        print('-'*30)

    logging.info(f"#Whole Beam: {len(hypos)}, #finished: ")
    logging.info('\n\n\n\n\n')
    for hypo in hypos:
        if not hypo.finished:
            logging.info(f"Not finished: {hypo}")
            continue
        logging.info(f"\n\n {hypo}")
        hypo.print_lattice()
    outputs = []
    """
    for unit in finished:
        logging.info(repr(unit))
        outputs.append(pprint(unit.token_full))
    """
    fname = render_name(doc_input_ids, beam_size, max_len,
                        param_sim_function['ngram_suffix'], param_sim_function['len_diff']) + '.pkl'
    with open(f"vizs/{fname}", 'wb') as fd:
        pickle.dump(hypos, fd)

    # score = eval_group_diversity(outputs)
    return outputs