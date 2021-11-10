
from src.recom_search.model.setup import tokenizer
import torch
import math
import logging
# from .util import pnum, tokenizer

import statistics

import random
import string
from typing import List
import numpy as np
from src.recom_search.model.bfs_util import NewHash
random.seed(2021)


def gen_rand_id(N=10):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))


def pprint(token_ids: List):
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


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
    def __init__(self, hash: NewHash, prob: float, token_idx: int, prev: List[str], prev_score: List,  min_len=10, finished=False, len_reward=0.0, master_node_uid=None) -> None:
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
        self.set_full()
        assert self.all_token_idx
        assert self.all_score
        assert self.length

        self.finished = finished
        self.min_len = min_len
        self.len_reward = len_reward
        self.has_finished()
        self.hash.set_node(self.uid, self)

    def has_finished(self):
        if (self.token_str.strip() == '.' or self.token_str.strip() == '</s>') and self.length >= self.min_len:
            self.finished = True
        else:
            self.finished = False

    def get_antecedent(self, step=None):
        antecedents = []

        prev = self.prev  # prev is a list
        _, prev = self.hash.retrieve_group_nodes(prev)
        cnt = 0
        while prev:

            antecedents += prev
            new_prev = []
            for p in prev:
                new_prev += p.prev
            _, new_prev = self.hash.retrieve_group_nodes(new_prev)
            # new_prev = list(set(new_prev))
            new_prev = [x for x in new_prev if x not in antecedents]
            prev = new_prev
            cnt += 1
            if step is not None and cnt >= step:
                break
        return antecedents

    def add_prev_node(self, node_id, score):
        """
        self: a b c d  a   f g
        node: a b c d  x y 

        """
        # check if self is the ancedant node of "node"
        if self in self.hash.retrieve_node(node_id).get_antecedent() or self.hash.find_root_node_uid(self.uid)  == self.hash.find_root_node_uid(node_id) or (node_id in self.prev):
            return

        self.prev.append(node_id)
        self.prev_score.append(score)

    def visualization(self):
        nodes, edges = {}, {}
        seen = {}

        def dfs(node: BeamNode):
            if not node:
                return
            node_uid = self.hash.find_root_node_uid(node.uid)
            if node_uid in seen:
                return
            seen[node_uid] = True
            node = self.hash.retrieve_node(node_uid)
            my_prev, my_prev_score = node.prev, node.prev_score
            for p, ps in zip(my_prev, my_prev_score):
                p_uid = self.hash.find_root_node_uid(p)
                p_node = self.hash.retrieve_node(p_uid)
                edge_info = {
                    'src': p_uid,
                    'tgt': node_uid,
                    'score': ps
                }
                edges[f"{p_uid}_{node_uid}"] = edge_info
                # edges.append(edge_info)

            nodes[node.uid] = {
                'uid': node_uid,
                'text': node.token_str,
                'tok_idx': node.token_idx
            }

            prevs = node.prev
            _, prevs = self.hash.retrieve_group_nodes(prevs)
            for p in prevs:
                dfs(p)
        dfs(self)
        return nodes, edges

    def print_lattice(self):
        raise NotImplementedError
        # DFS to discover nodes, if a node is seen and discovered again, it's the start of a span
        # key is a node, value is the latest path to this node from root.
        seen = {}
        recomb_units = []

        def dfs(node, par_nodes):
            if not node:
                return
            if node.uid in seen:
                last_path = seen[node.uid]
                cur_path = par_nodes
                last_path_tokens = [x.token_idx for x in last_path]
                cur_path_tokens = [x.token_idx for x in cur_path]
                # order of last_path_tokens and cur_path_tokens: [root, ->, ...]
                shared_prefix_len, _ = find_prefix(
                    last_path_tokens, cur_path_tokens)
                last = last_path_tokens[shared_prefix_len:][::-1]
                newer = cur_path_tokens[shared_prefix_len:][::-1]
                shared_tokens = last_path_tokens[:shared_prefix_len][::-1]
                logging.info(
                    f"\n======{tokenizer.decode(last)} || {tokenizer.decode(shared_tokens)} \n-----{tokenizer.decode(newer)} || {tokenizer.decode(shared_tokens)} ")
                recomb_units.append([last, newer])
                seen[node.uid] = par_nodes
                return
            seen[node.uid] = par_nodes
            prevs = node.prev
            for p in prevs:
                dfs(p, par_nodes + [node])

        dfs(self, [])
        logging.info(
            f"There are {len(recomb_units)} recomb units in this case.")

    def set_full(self):
        """
        Everytime a substructure is modified, we need to update the tokens and scores
        """
        tokens = [self.token_idx]
        scores = [self.score]
        prev = self.prev
        while prev:
            prev = prev[0]
            prev = self.hash.retrieve_node(prev)
            tokens.append(prev.token_idx)
            scores.append(prev.score)
            prev = prev.prev
        self.all_score = scores[::-1]
        self.all_token_idx = tokens[::-1]
        self.length = len(tokens)

    def get_tokens_str(self):
        out = [self.token_str]
        prev = self.prev
        while prev:
            prev = prev[0]
            prev = self.hash.retrieve_node(prev)
            out.append(prev.token_str)
            prev = prev.prev
        out = out[::-1]
        return '-'.join(out)

    def get_token_idx_as_input(self):
        tokens = self.all_token_idx
        dec_prefix = torch.tensor([tokens], dtype=torch.long)
        return dec_prefix

    def get_score_sum(self):
        all_score = self.all_score
        return sum(all_score) + self.len_reward * len(all_score)

    def get_score_avg(self):
        return statistics.mean(self.all_score)

    def __repr__(self) -> str:
        return self.get_tokens_str()

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
                raise Exception("Not found!")
        matched_suffix_node_ids = prev[0][:-1]
        prev = prev[0]
        while prev:
            node_uid = prev[0]
            node = self.hash.retrieve_node(node_uid)
            reversed_tokens.append(node.token_idx)
            prev = node.prev

        return reversed_tokens[::-1], matched_suffix_node_ids
