from src.recom_search.model.beam_state import BeamNode

import math
import scipy
import random
import logging
import spacy

nlp = spacy.load("en_core_web_sm")
all_stopwords = spacy.lang.en.stop_words.STOP_WORDS


class DeployHeu():
    def __init__(self, heu_config) -> None:
        self.heu_config = heu_config

    def run(self, cur_node, prev_len, prob_distb):
        prob_distb = prob_distb.squeeze().cpu().numpy()
        seq_score = self.heuristic_seq_score(
            cur_node) * self.heu_config['heu_seq_score']
        len_rwd = self.heuristic_seq_score_len_rwd(
            prev_len) * self.heu_config['heu_seq_score_len_rwd']
        ent = self.heuristic_ent(prob_distb) * self.heu_config['heu_ent']
        pos_bias = self.heuristic_position(
            prev_len) * self.heu_config['heu_pos']

        good_word_reward = self.heuristic_token(
            cur_node) * self.heu_config['heu_word']
        if random.random() < 0.01:
            logging.info(
                f"Heuristic breakdown: {seq_score}\t{len_rwd}\t{ent}\t{pos_bias}\t{good_word_reward}")
        return seq_score + len_rwd + ent + pos_bias + good_word_reward

    def heuristic_seq_score(self, cur_node: BeamNode) -> float:
        """
        """
        # scores = cur_node.get_path_sample()
        # s = sum(scores)
        # assert s <= 0
        return 0

    def heuristic_seq_score_len_rwd(self, prev_len):
        return prev_len

    def heuristic_position(self, prev_len):
        # return math.exp(-prev_len)
        return max((10 - prev_len)/10, 0)

    def heuristic_ent(self, prob_distb):
        ent = scipy.stats.entropy(prob_distb)
        return ent

    def heuristic_token(self, cur_node):
        s = cur_node.token_str.strip()
        if s in all_stopwords:
            return 0
        else:
            return 1
