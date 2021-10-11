import statistics
import math
from collections import defaultdict
from src.recom_search.model.beam_state import BeamNode


def convert_seq_score(seq, inp_score):
    # seq: List[List[int]]
    # inp_score: seq_len, batch,  vocab_size
    batch_size = len(seq)
    seq_len = len(seq[0])
    converted_scores = []
    for b in range(batch_size):
        tmp = []
        for t in range(1, seq_len):
            token_idx = seq[b][t]
            score = inp_score[t-1][b][token_idx]
            tmp.append(score)
        converted_scores.append(tmp)
    return converted_scores


def truncate_sequence(seq):
    l = len(seq)
    trunc_seq = []
    for i in range(l):
        tmp_seq = []
        cur_seq = seq[i]
        cur_seq_len = len(cur_seq)
        j = 0
        while j < cur_seq_len:
            tok = cur_seq[j]
            if tok == 2 and j > 0:
                break
            tmp_seq.append(tok)
            j += 1
        trunc_seq.append(tmp_seq)
    return trunc_seq


"""
def truncate_sequence(seq, seq_score):
    assert len(seq) == len(seq_score)
    l = len(seq)
    trunc_seq, trunc_seq_score = [], []
    for i in range(l):
        tmp_seq, tmp_seq_score = [], []
        cur_seq = seq[i]
        cur_seq_score = seq_score[i]
        cur_seq_len  = len(cur_seq)
        j = 0
        while j < cur_seq_len:
            tok = cur_seq[j]
            sco = cur_seq_score[j]
            if tok == 2 and j > 0:
                break
            tmp_seq.append(tok)
            tmp_seq_score.append(sco)
            j += 1
        trunc_seq.append(tmp_seq)
        trunc_seq_score.append(tmp_seq_score)
    return trunc_seq, trunc_seq_score
"""


def construct_trees(seq):
    # construct the graph/tree from a group of outputs
    d = {}
    ends = []
    min_len = 1000
    for s in seq:
        prev_tokens = []
        for x in s:

            key = "_".join(prev_tokens + [str(x)])
            if key in d:
                prev_tokens.append(str(x))
                continue
            # a new pattern
            father_key = "_".join(prev_tokens)
            if father_key:
                father = d[father_key]
                tmp = BeamNode(1, x, prev=[father], prev_score=[father.score])
            else:
                tmp = BeamNode(1, x, prev=[], prev_score=[])
            d[key] = tmp
            prev_tokens.append(str(x))
        min_len = min(min_len, len(prev_tokens))
        ends.append(tmp)
    # branching factor
    """
    nodes_in_len_bucket = [0 for _ in range(30)]
    for key, value in d.items():
        l = value.length
        nodes_in_len_bucket[l] += 1
    nodes_in_len_bucket = [x for x in nodes_in_len_bucket if x != 0]
    nodes_in_len_bucket = nodes_in_len_bucket[:min_len]
    effective_len = len(nodes_in_len_bucket)
    bucket = []
    for i in range(effective_len-1):
        prev, nxt = nodes_in_len_bucket[i], nodes_in_len_bucket[i+1]
        factor = nxt/prev
        bucket.append(factor)
    quants = statistics.quantiles(bucket, n=10)
    print(quants)
    """
    return ends
    return ends, quants
