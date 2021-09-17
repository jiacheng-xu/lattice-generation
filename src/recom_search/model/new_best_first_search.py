from _typeshed import StrPath
from collections import defaultdict
from typing import List
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

def generate_merge(start_seed, hash, heap,  doc_input_ids, model, param_sim_function, max_len, explore_steps):
    # try to extend the start_seed for explore_steps steps. if there is a mergable match, do that match, else, finish the generation
    

def new_best_first_search(doc_input_ids, model, param_sim_function, eos_token_id=21, explore_steps=10, max_len=20, num_return_hypo=100, debug: bool = False):
    total_calls = 0
    explored_cnt = 0
    init_seed = NewBeamState(token_idx=eos_token_id, prev=[])
    gen_hash = NewHash
    h = []
    heapq.heappush(h, (-init_seed.prob, init_seed))
    while h:
        s = heapq.heappop(h)
        explored_cnt += 1
        prob, seed = s
        generate_merge(start_seed=seed, )
