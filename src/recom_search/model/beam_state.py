
from src.recom_search.model.util import gen_rand_id
# from src.recom_search.model.exec_setup import tokenizer
import torch
import math
import logging

import random
import string
from typing import List
import numpy as np
from src.recom_search.model.bfs_util import HashObject
random.seed(2021)



# def pprint(token_ids: List):
#     return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def find_prefix(seq_a, seq_b):
    pointer_a, pointer_b = 0, 0
    while pointer_a < len(seq_a) and pointer_b < len(seq_b):
        a = seq_a[pointer_a]
        b = seq_b[pointer_b]
        if a != b:
            return [pointer_a, pointer_b]
        else:
            pointer_a += 1
            pointer_b += 1
    return [pointer_a, pointer_b]


def find_suffix(seq_a, seq_b):
    pointer_a, pointer_b = len(seq_a)-1, len(seq_b) - 1
    while pointer_a >= 0 and pointer_b >= 0:
        a = seq_a[pointer_a]
        b = seq_b[pointer_b]
        if a != b:
            return [pointer_a, pointer_b]
        else:
            pointer_a -= 1
            pointer_b -= 1
    return [pointer_a, pointer_b]



class BeamNode():
    def __init__(self, hash: HashObject, prob: float, token_idx: int, prev: List[str], prev_score: List, min_len=10, finished=False, len_reward=0.0, master_node_uid=None) -> None:
        self.hash = hash
        self.uid = gen_rand_id()
        self.prob = prob
        self.score = math.log(prob)
        self.prev = prev      # prev is always sorted where top1 has highest score
        self.prev_score = prev_score
        self.token_idx = token_idx
        # print(self.token_idx)
        self.token_str = tokenizer.decode(
            self.token_idx, skip_special_tokens=False) if tokenizer else f"{token_idx}"
        self.master_node_uid =master_node_uid or self.uid
        # self.set_full()
        assert self.all_token_idx
        assert self.all_score
        assert self.length

        self.finished = finished
        self.min_len = min_len
        self.len_reward = len_reward
        self.has_finished()
        self.hash.set_node(self.uid, self)


    def get_tokens_match_suffix(self, inp_suffix_tokens: List[int]):
        """
        suffix_tokens is the target suffix to match
        greedily pick the prev[0] for rest of sequence
        possible improvement, return all possible and do ANY for heuristic math
        """
        reversed_tokens = []
        suffix_tokens = inp_suffix_tokens.copy()
        
        prev = [[self.uid]]
        while prev and suffix_tokens:
            last_target_token_idx = suffix_tokens.pop(-1)
            # print('-----')
            # print(tokenizer.decode(last_target_token_idx))
            new_prev = []
            for list_p in prev:
                # p is a list of UIDs, the last one is the latest one
                p = list_p[-1]
                node = self.hash.retrieve_node(p)
                # print(node.token_str)
                token = node.token_idx
                if token == last_target_token_idx:
                    # _, tmp_prev = self.hash.retrieve_group_nodes(node.prev)
                    new_prev += [list_p + [x] for x in node.prev] 

            reversed_tokens.append(last_target_token_idx)
            prev = new_prev
            if not prev:
                raise ValueError("Not found!")
        matched_suffix_node_ids = prev[0][:-1]
        prev = prev[0]
        while prev:
            node_uid = prev[0]
            node = self.hash.retrieve_node(node_uid)
            reversed_tokens.append(node.token_idx)
            prev = node.prev

        return reversed_tokens[::-1], matched_suffix_node_ids
