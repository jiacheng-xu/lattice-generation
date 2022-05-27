from typing import List
from src.recom_search.model.util import gen_rand_id
import random,math
from src.recom_search.model.exec_setup import tokenizer
from abc import ABC, abstractmethod
import torch
random.seed(2021)
from statistics import mean

class BeamNode(ABC):
    def __init__(self, prob: float, token_idx: int, prev: List, prev_score: List, min_len:int=10, finished:bool=False) -> None:
        self.uid = gen_rand_id()
        self.prob = prob
        self.score = math.log(prob)
        self.prev = prev
        self.prev_score = prev_score
        self.token_idx = token_idx

        self.token_str = tokenizer.decode(
            self.token_idx, skip_special_tokens=False) if tokenizer else f"{token_idx}"

        self.finished = finished
        self.min_len = min_len

    def get_repr(self, inp):
        """
        For BeamNodeFull, we need to use the hash function to retrieve the node from a dictionary; For BeamNodeEz, it's naturally a node already.
        """
        if isinstance(inp, BeamNode):
            return inp
        else:
            return self.hash.retrieve_node(inp)

    def has_finished(self):
        if (self.token_str.strip() == '.' or self.token_str.strip() == '</s>') and self.length >= self.min_len:
            self.finished = True
        else:
            self.finished = False

    def __len__(self):
        return self.length

    def set_canonical_path(self):
        """
        To get the canonical path, we will recursively vist the first node of all previous nodes.
        """
        tokens = [self.token_idx]
        scores = [self.score]
        prevs = self.prev    # prev is a list
        while prevs:
            prev = prevs[0]
            prev_repr = self.get_repr(prev)
            tokens.append(prev_repr.token_idx)
            scores.append(prev_repr.score)
            prevs = prev_repr.prev        # update pointer
        self.all_score = scores[::-1]
        self.all_token_idx = tokens[::-1]
        self.length = len(tokens)

    def get_canonical_str(self, split_tok='-')->str:
        out = [self.token_str]
        prevs = self.prev
        while prevs:
            prev = prevs[0]
            prev_repr = self.get_repr(prev)

            out.append(prev_repr.token_str)
            prevs = prev_repr.prev
        out = out[::-1]
        return split_tok.join(out)

    def get_token_idx_as_input(self):
        tokens = self.all_token_idx
        dec_prefix = torch.tensor([tokens], dtype=torch.long)
        return dec_prefix
    def get_score_sum(self):
        all_score = self.all_score
        return sum(all_score)

    def get_score_avg(self):
        return mean(self.all_score)

    def __repr__(self) -> str:
        return self.get_tokens_str()

