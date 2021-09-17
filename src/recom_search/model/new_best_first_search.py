
from collections import defaultdict
from typing import List

from transformers.tokenization_utils_base import INIT_TOKENIZER_DOCSTRING
from src.recom_search.model.recomb_proto import new_merge_core, similarity_heuristic
from src.recom_search.model.util import run_inference_step
from src.recom_search.model.beam_state import NewBeamState

import heapq


def construct_init_pad_sent(eos_token_id, max_len):
    cnt = 0
    init_seed = NewBeamState(token_idx=eos_token_id, prev=[])
    pointer = init_seed
    # we don't need this


class NewHash():
    def __init__(self, ngram: int = 5) -> None:
        self.data = defaultdict(list)
        self.ngram = ngram

    def const_key(self, token_ids):
        tokens = token_ids[-self.ngram:]
        k = "_".join(tokens)
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
import torch
def generate_merge(start_seed, hash:NewHash, heap,  doc_input_ids, model, param_sim_function, max_len, explore_steps,k_best):
    # try to extend the start_seed for explore_steps steps. if there is a mergable match, do that match, else, finish the generation
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
        cur_dec_input_ids = pointer.all_token_idx
        dec_prefix = pointer.get_token_idx_as_input()
        _, output_prob, _, _ = run_inference_step(
            model, doc_input_ids, decoder_input_ids=dec_prefix, device=doc_input_ids.device, output_dec_hid=False)
        values, indices = torch.topk(output_prob, k=k_best)
        values = values[0].tolist()
        indices = indices[0].tolist()
        # is top1 in hash?
        if cur_len < target_steps and cur_len >= hash.ngram:
            retrieved = hash.query(cur_dec_input_ids + [indices[0]])
            ngram = (cur_dec_input_ids + [indices[0]])[-ngram_suffix:]
            if retrieved:   # are there possible hash there?
                for candidate_pair in retrieved:
                    span_beg, span_end = candidate_pair
                    one_match_path_token_ids = span_end.get_tokens_match_suffix(ngram)
                    flag = similarity_heuristic(one_match_path_token_ids, pointer.all_token_idx, ngram_suffix, len_diff)
                    if flag:
                        flag_merge = True
                        break
                if flag_merge:
                    new_merge_core(span_end,pointer)
                    break

        # add stuff to heap
        top_ranked_state = None
        for v,i in zip(values,indices):
            tmp_state = NewBeamState(prob=v, token_idx = i, prev=[pointer])
            if top_ranked_state == None:
                top_ranked_state = tmp_state
            heapq.heappush(heap, (-v, tmp_state))
        pointer = top_ranked_state
        if pointer.finished:
            break
        cur_len += 1
    if flag_merge:
        return None
    else:
        return pointer
    


    
    

def new_best_first_search(doc_input_ids, model, param_sim_function, eos_token_id=21, explore_steps=10, max_len=20, k_best = 5, num_return_hypo=100, debug: bool = False):
    total_calls = 0
    explored_cnt = 0
    outputs = []
    init_seed = NewBeamState(token_idx=eos_token_id, prev=[])
    gen_hash = NewHash()
    h = []
    heapq.heappush(h, (-init_seed.prob, init_seed))
    while h:
        s = heapq.heappop(h)
        explored_cnt += 1
        prob, seed = s
        output = generate_merge(start_seed=seed, hash=gen_hash, heap=h,doc_input_ids=doc_input_ids, model=model,param_sim_function=param_sim_function,max_len=max_len,explore_steps=explore_steps,k_best=k_best)
        if output:
            outputs.append(output)
        if explored_cnt >= num_return_hypo:
            break
    return outputs
