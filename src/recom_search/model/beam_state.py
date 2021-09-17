
import torch
import math
import logging
# from .util import pnum, tokenizer
from src.recom_search.model.util import pnum, tokenizer
import statistics

import random
import string
from typing import List

def gen_rand_id(N=5):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

def pprint(token_ids:List):
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

class NewBeamState():
    def __init__(self, prob:float, token_idx:int, prev:List, min_len=10, finished=False, len_reward=0.0) -> None:
        self.uid = gen_rand_id()
        self.prob = prob
        self.score = math.log(prob)
        self.prev = prev      # prev is always sorted where top1 has highest score
        self.token_idx = token_idx
        self.token_str = tokenizer.decode(self.token_idx) if tokenizer else f"{token_idx}"

        self.set_full()
        assert self.all_token_idx
        assert self.all_score
        assert self.length
        # self.all_token_idx = 

        self.finished = finished
        self.min_len = min_len
        self.len_reward = len_reward
        self.has_finished()
    
    def has_finished(self):
        if self.token_str.strip() == '.' and self.length >= self.min_len:
            self.finished = True
        else:
            self.finished = False

    def add_prev_node(self, node):
        """
        self: a b c d  a   f g
        node: a b c d  x y 

        """
        self.prev.append(node)
        # sort

    def visualization(self):
        nodes, edges = [], []
        seen = {}
        def dfs(node: NewBeamState, par_nodes):
            if not node:
                return
            
            if par_nodes:
                edge_info = {
                    'src':node.uid,
                    'tgt':par_nodes[-1].uid,
                    'seen':False
                }
                
            if node.uid in seen:
                edge_info['seen'] = True
                edges.append(edge_info)
                return
            seen[node.uid] = True
            nodes.append({
                'uid':node.uid,
                'text':node.token_str,
                'score':node.score
            })
            if par_nodes:
                edges.append(edge_info)
            prevs = node.prev
            for p in prevs:
                dfs(p, par_nodes + [node])
        dfs(self, [])
        return nodes, edges
    def print_lattice(self):
        # DFS to discover nodes, if a node is seen and discovered again, it's the start of a span
        seen = {}   # key is a node, value is the latest path to this node from root. 
        recomb_units = []
        def dfs(node, par_nodes):
            if not node:
                return
            if node.uid in seen:
                last_path = seen[node.uid]
                cur_path = par_nodes
                last_path_tokens = [ x.token_idx for x in last_path]
                cur_path_tokens = [ x.token_idx for x in cur_path]
                # order of last_path_tokens and cur_path_tokens: [root, ->, ...]
                shared_prefix_len, _ = find_prefix(last_path_tokens, cur_path_tokens)
                last = last_path_tokens[shared_prefix_len:][::-1]
                newer =  cur_path_tokens[shared_prefix_len:][::-1]
                shared_tokens = last_path_tokens[:shared_prefix_len][::-1]
                logging.info(f"\n======{tokenizer.decode(last)} || {tokenizer.decode(shared_tokens)} \n-----{tokenizer.decode(newer)} || {tokenizer.decode(shared_tokens)} ")
                recomb_units.append([  last,newer ]  )
                seen[node.uid] = par_nodes
                return
            seen[node.uid] = par_nodes
            prevs = node.prev
            for p in prevs:
                dfs(p, par_nodes + [node])
        
        dfs(self, [])
        logging.info(f"There are {len(recomb_units)} recomb units in this case.")

    def set_full(self):
        """
        Everytime a substructure is modified, we need to update the tokens and scores
        """
        tokens = [self.token_idx]
        scores = [self.score]
        prev = self.prev
        while prev:
            prev = prev[0]
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
            out.append(prev.token_str)
            prev = prev.prev
        out = out[::-1]
        return '<-'.join(out)


    def get_token_idx_as_input(self):
        tokens = self.all_token_idx
        dec_prefix = torch.tensor([tokens], dtype=torch.long)
        return dec_prefix

    def _get_length(self):
        l = 1
        prev = self.prev
        while prev:
            prev = prev[0]
            l+=1
            prev = prev.prev
        return l
    
    def get_score_sum(self):
        all_score = self.all_score
        return sum(all_score) + self.len_reward * len(all_score)

    def get_score_avg(self):
        return statistics.mean(self.all_score)
    def __repr__(self) -> str:
        return self.get_tokens_str()

    def get_tokens_match_suffix(self, suffix_tokens:List[int]):
        reversed_tokens = []

        prev = [self]
        while prev or suffix_tokens:
            last_target_token_idx = suffix_tokens.pop(-1)
            for p in prev:
                token = p.token_idx
                if token == last_target_token_idx:
                    reversed_tokens.append(token)
                    prev = p.prev
                    continue
            raise Exception("Not found!")
        while prev:
            prev = prev[0]
            reversed_tokens.append(prev.token_idx)
            prev = prev.prev
        return reversed_tokens[::-1]

    
class BeamState(object):
    def __init__(self, cur_idx_in_distb, prob_distrib, token_id_distb,  prev=[], min_len=10, finished=False, len_reward=0.0) -> None:
        super().__init__()
        self.uid = gen_rand_id()
        self.score = math.log(prob_distrib[cur_idx_in_distb])
        self.prob = prob_distrib[cur_idx_in_distb]
        self.token = token_id_distb[cur_idx_in_distb]  # token
        self.token_str = tokenizer.decode(self.token) if tokenizer else "[empty]"

        
        self.peer_prob = prob_distrib  # rank in the current peer
        self.peer_token_id = token_id_distb

        self.prev = prev

        self.token_full = self.__get_tokens()
        self.token_str_full = [tokenizer.convert_ids_to_tokens(x) for x in self.token_full]
        self.score_full = self.__get_scores()
        self.finished = finished
        self.min_len = min_len
        self.len_reward = len_reward
        self.has_finished()
        if prev == []:
            self.len = 1
        else:
            self.len = self.prev.len + 1
        self.merge = []

    def has_finished(self):
        if self.token_str.strip() == '.' and len(self.token_full) >= self.min_len:
            self.finished = True
        else:
            self.finished = False

    def __get_tokens(self):
        tokens = [self.token]
        prev = self.prev
        while prev:
            tokens.append(prev.token)
            prev = prev.prev
        return tokens[::-1]

    def __get_scores(self):
        scores = [self.score]
        prev = self.prev
        while prev:
            scores.append(prev.score)
            prev = prev.prev
        return scores[::-1]

    def get_tokens_str(self):
        tokens = self.token_full
        tokens_str = [tokenizer.convert_ids_to_tokens(x) for x in tokens]
        return tokens_str


    def get_tokens_as_input(self):
        tokens = self.token_full
        dec_prefix = torch.tensor([tokens], dtype=torch.long)
        return dec_prefix

    def get_avg_score(self):
        return statistics.mean(self.score_full)

    def get_score_sum(self):
        all_scores = self.score_full
        return sum(all_scores) + self.len_reward * len(all_scores)

    def get_partial_score(self, start_idx, end_idx):
        all_scores = self.score_full
        partial_scores = all_scores[start_idx:end_idx]
        return sum(partial_scores) + self.len_reward * len(partial_scores)

    def get_complete_repr(self, k=2):
        tokens = [self.token]
        probs = [self.prob]
        top_k_tokens = [self.peer_token_id[:k]]
        top_k_probs = [self.peer_prob[:k]]
        prev = self.prev
        while prev:
            tokens.append(prev.token)
            probs.append(prev.prob)
            top_k_tokens.append(prev.peer_token_id[:k])
            top_k_probs.append(prev.peer_prob[:k])
            prev = prev.prev
        tokens = tokens[::-1]
        probs = probs[::-1]
        tokens_str = [tokenizer.convert_ids_to_tokens(x) for x in tokens]
        top_k_tokens = top_k_tokens[::-1]
        top_k_tokens_str = [[tokenizer.convert_ids_to_tokens(
            tk) for tk in x] for x in top_k_tokens]
        top_k_probs = top_k_probs[::-1]

        header = ['T', 'TP'] + [f"[{idx}]" for idx in range(k)]

        rows = [[] for _ in range(len(header))]
        rows[0] = [header[0]] + \
            ['{:15d}'.format(x) for x in range(len(tokens_str))]
        rows[1] = [header[1]] + \
            ["{:3s} {:10s}".format(pnum(y), x)
             for x, y in zip(tokens_str, probs)]
        for idx in range(k):
            for x, y in zip(top_k_tokens_str, top_k_probs):
                rows[2+idx].append("{:3s} {:10s}".format(pnum(y[idx]), x[idx]))
            rows[2+idx] = [header[2+idx]] + rows[2+idx]

        pointer = 0
        cache = [[] for _ in range(len(header))]
        while pointer < len(tokens):
            for jdx, r in enumerate(rows):
                cache[jdx].append(r[pointer])
            if pointer % 8 == 7:
                for c in cache:
                    logging.info("\t".join(c))
                cache = [[] for _ in range(len(header))]
            pointer += 1
        if cache[0] != []:
            for c in cache:
                logging.info("\t".join(c))

    def get_simple_repr(self):
        tokens = [self.token]
        probs = [self.prob]

        prev = self.prev
        while prev:
            tokens.append(prev.token)
            probs.append(prev.prob)
            prev = prev.prev
        tokens = tokens[::-1]
        probs = probs[::-1]
        tokens_str = [tokenizer.convert_ids_to_tokens(x) for x in tokens]

        header = ['T', 'TP']

        rows = [[] for _ in range(len(header))]
        rows[0] = [header[0]] + \
            ['{:12d}'.format(x) for x in range(len(tokens_str))]
        rows[1] = [header[1]] + \
            ["{:3s} {:8s}".format(pnum(y), x)
             for x, y in zip(tokens_str, probs)]
        pointer = 0
        cache = [[] for _ in range(len(header))]
        while pointer < len(tokens):
            for jdx, r in enumerate(rows):
                cache[jdx].append(r[pointer])
            if pointer % 10 == 9:
                for c in cache:
                    logging.info("\t".join(c))
                cache = [[] for _ in range(len(header))]
            pointer += 1
        if cache[0] != []:
            for c in cache:
                logging.info("\t".join(c))

    def get_ancestor_uid(self):
        UIDs = []
        prev = self.prev
        while prev:
            UIDs.append(prev.uid)
            prev = prev.prev
        return UIDs


    def __repr__(self):
        # retrieve merges
        merge_str = []
        if self.merge:
            for meg in self.merge:
                sp, sp_option = meg
                merge_str.append(f"-Option: {sp} <-> {sp_option}")
        merge_str = "\n".join(merge_str)
        init_str = f"Avg Score: {pnum(self.get_avg_score())}\tSum: {pnum(self.get_score_sum())}\tTokens: {pprint(self.token_full)}\n"
        return init_str + merge_str + '\n'

    def add_merge_record(self, my_span, merged_span, merge_histroy_of_target):
        # logging.info(f"MERGE!")
        logging.info(f"WinSpan: {my_span}\nLostSpan: {merged_span}")
        merged_span.prefix_node = my_span.prefix_node
        merged_span.suffix_node = my_span.suffix_node
        self.merge.append([my_span, merged_span])
        self.merge += merge_histroy_of_target
