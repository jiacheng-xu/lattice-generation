import torch
import math
import logging
from .util import pnum, tokenizer
import statistics

import random
import string


def gen_rand_id(N=5):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))


class BeamState(object):
    def __init__(self, cur_idx_in_distb, prob_distrib, token_id_distb,  prev=[], min_len=10, finished=False, len_reward=0.0) -> None:
        super().__init__()
        self.uid = gen_rand_id()
        self.score = math.log(prob_distrib[cur_idx_in_distb])
        self.prob = prob_distrib[cur_idx_in_distb]
        self.token = token_id_distb[cur_idx_in_distb]  # token
        self.token_str = tokenizer.decode(
            self.token) if tokenizer else "[empty]"

        self.peer_prob = prob_distrib  # rank in the current peer
        self.peer_token_id = token_id_distb

        self.prev = prev

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
        if self.token_str.strip() == '.' and len(self.get_tokens()) >= self.min_len:
            self.finished = True
        else:
            self.finished = False

    def get_tokens(self):
        tokens = [self.token]
        prev = self.prev
        while prev:
            tokens.append(prev.token)
            prev = prev.prev
        return tokens[::-1]

    def get_tokens_str(self):
        tokens = self.get_tokens()
        tokens_str = [tokenizer.convert_ids_to_tokens(x) for x in tokens]
        return tokens_str

    def get_scores(self):
        scores = [self.score]
        prev = self.prev
        while prev:
            scores.append(prev.score)
            prev = prev.prev
        return scores[::-1]

    def get_tokens_as_input(self):
        tokens = self.get_tokens()
        dec_prefix = torch.tensor([tokens], dtype=torch.long)
        return dec_prefix

    def get_avg_score(self):
        return statistics.mean(self.get_scores())

    def get_score_sum(self):
        all_scores = self.get_scores()
        return sum(all_scores) + self.len_reward * len(all_scores)

    def get_partial_score(self, start_idx, end_idx):
        all_scores = self.get_scores()
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

    def get_output_str(self):
        return tokenizer.decode(self.get_tokens(), skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def __repr__(self):
        # retrieve merges
        merge_str = []
        if self.merge:
            for meg in self.merge:
                sp, sp_option = meg
                merge_str.append(f"-Option: {sp} <-> {sp_option}")
        merge_str = "\n".join(merge_str)
        init_str = f"Avg Score: {pnum(self.get_avg_score())}\tSum: {pnum(self.get_score_sum())}\nTokens: {self.get_output_str()}\n"
        return init_str + merge_str

    def add_merge_record(self, my_span, merged_span, merge_histroy_of_target):
        # logging.info(f"MERGE!")
        logging.info(f"WinSpan: {my_span}\nLostSpan: {merged_span}")
        self.merge.append([my_span, merged_span])
        self.merge += merge_histroy_of_target
