import pickle
from collections import defaultdict
from re import split
import statistics

import argparse
from typing import List
import math
from scipy.stats import entropy
from numpy.lib.utils import who
from scipy.stats.morestats import wilcoxon
from transformers.utils.dummy_tokenizers_objects import LEDTokenizerFast

from src.recom_search.evaluation.vis import viz_result
from .util import *
from src.recom_search.model.beam_state import BeamState, pprint
# from recombination_prototype import eval_group_diversity

LEN_DIFF = 8  # max diff of two string
MAX_STEP = 30  # max gen steps
BS = 20
NGRAM_SUF = 3  # last NGRAM_SUF tokens match
RWD_LEN = 0.08  # score = \sum RWD_len + log p(y)
logging.info(f"BS:{BS} SUFFIX:{NGRAM_SUF} MAX_STEP:{MAX_STEP}")

debug = False

model_name = 'sshleifer/distilbart-xsum-12-6'
# tokenizer = BartTokenizer.from_pretrained(model_name)


def sublist(lst1, lst2):
    return set(lst1) <= set(lst2)


class GenHash():
    def __init__(self, ngram: int = 5, back_track_steps: int = 2) -> None:
        self.data = defaultdict(list)
        self.back_step = back_track_steps
        self.ngram = ngram

    def query(self, token_ids: List[int]):
        return None
        if len(token_ids) < self.ngram:
            return []
        l = len(token_ids)  # original len
        token_ids = token_ids[-self.ngram:]
        k = "_".join([str(x) for x in token_ids])
        # l = len(token_ids)
        if k in self.data:
            outputs = []
            vs = self.data[k]
            for bt in range(self.back_step):
                if l-bt in vs:
                    outputs += vs[l-bt]
            if not outputs:
                return []
            return outputs
        return []

    def add(self, tokens: List, beam_node):
        l = len(tokens)
        if l < self.ngram:
            return
        token_ids = tokens[-self.ngram:]
        k = "_".join([str(x) for x in token_ids])
        if l in self.data[k]:
            self.data[k][l].append(beam_node)
        else:
            self.data[k] = {l: [beam_node]}

    def delete_successor(self):
        pass

def fake_model_output(vocab_size=20, k=BS):
    output = torch.rand(vocab_size) * 20
    softmax_scores = torch.nn.functional.softmax(output)
    return torch.topk(softmax_scores, k=k)

class NewSpan():
    def __init__(self, left, right, score) -> None:
        self.left = left    # the left end node
        self.right = right  # the right end node
        self.edges = []
        self.score = score  # the best-scored path's score in this span

class Span():
    def __init__(self, tokens, token_strs, score: float, prefix_node=None, suffix_node=None, prefix=[], suffix=[]) -> None:
        # [A: [token, token, token], [score, score, score], B: [token, token], [score, score], .....]
        # prefix if provided
        # we might want something like model repr in future
        self.tokens = tokens
        self.score = score
        self.prefix = prefix
        self.suffix = suffix
        self.token_strs = token_strs
        self.prefix_node = prefix_node
        self.suffix_node = suffix_node

    def __repr__(self) -> str:
        return f"Score: {pnum(self.score)}\n{tokenizer.decode(self.prefix)} [{tokenizer.decode(self.tokens)}] {tokenizer.decode(self.suffix)}"


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



def similarity_heuristic(a_tokens, b_tokens, ngram_suffix, len_diff) -> bool:

    if len(a_tokens) > ngram_suffix and len(b_tokens) > ngram_suffix:
        if a_tokens[-ngram_suffix:] == b_tokens[-ngram_suffix:]:
            logging.debug(f"Stage 1: Suffix match SUCCESS")
        else:
            return False
    else:
        return False

    # Stage 2: length
    if abs(len(a_tokens) - len(b_tokens)) < len_diff:
        logging.debug(f"Stage 2: Len Diff SUCCESS")
    else:
        # logging.debug(f"Stage 2: Len Diff FAIL")
        return False
    return True


def get_beam_from_past(end_beam, t):
    nodes = []
    point = end_beam
    while point:
        nodes.append(point)
        point = point.prev
    nodes = nodes[::-1]
    return nodes[t]

def new_merge_core(beam_par, beam_drop):
    logging.debug(beam_par.all_token_idx)
    logging.debug(beam_drop.all_token_idx)
    # when does their suffix starts to differ?
    pointer_par = beam_par
    pointer_drop = beam_drop
    # we just assume they share a same suffix
    par_paths = [pointer_par]
    # beam_drop is treated as a single line
    prev_par_paths = par_paths
    prev_pointer_drop = beam_drop
    while pointer_drop and par_paths:
        
        next_par_paths = []
        for par_path in par_paths:
            if pointer_drop.token_idx == par_path.token_idx:
                next_par_paths += par_path.prev  
        if next_par_paths:
            prev_pointer_drop = pointer_drop
            pointer_drop = pointer_drop.prev[0]
            prev_par_paths = par_paths
            par_paths = next_par_paths
            
        else:
            break    # suffix match end
    # pointer_drop is the first token that differs
    # par_paths is the first threads differs
    # prev_par_paths is the last match
    # add pointer_drop to prev_par_paths 's prev
    for path in prev_par_paths:
        path.add_prev_node(pointer_drop)
    beam_par.print_lattice()
    return beam_par
    # go leftward to end of prev_par_paths, get all nodes
    # go leftward to end of 

def merge_compare(beam_a, beam_b, merge_to_a: bool = False, ngram_suffix: int = 5, len_diff: int = 5):
    # we assume we are matching the suffix of a and b although their length can be different
    # try to merge a -> b
    # Step 1: suffix match
    a_tokens = beam_a.all_token_idx
    b_tokens = beam_b.all_token_idx
    flag = similarity_heuristic(a_tokens, b_tokens, ngram_suffix, len_diff)
    if not flag:
        return [beam_a, beam_b]

    a_tokens_str = beam_a.token_str_full
    b_tokens_str = beam_b.token_str_full

    # Stage 3: let's merge!
    if a_tokens == b_tokens:
        return [beam_a, None]
        score_a = beam_a.get_score_sum()
        score_b = beam_b.get_score_sum()
        span_a = Span(a_tokens, a_tokens_str, score=score_a,
                      prefix=[], suffix=a_tokens)
        span_b = Span(b_tokens, b_tokens_str, score=score_b,
                      prefix=[], suffix=b_tokens)
        if merge_to_a or score_a > score_b:
            beam_a.add_merge_record(span_a, span_b, beam_b.merge)
            return [beam_a, None]
        else:
            beam_b.add_merge_record(span_b, span_a, beam_a.merge)
            return [None, beam_b]

    prefix_a, prefix_b = find_prefix(a_tokens, b_tokens)
    suf_a, suf_b = find_suffix(a_tokens, b_tokens)

    diff_a = a_tokens[prefix_a:suf_a+1]
    diff_b = b_tokens[prefix_b:suf_b+1]

    score_a, score_b = beam_a.get_partial_score(
        prefix_a, suf_a+1), beam_b.get_partial_score(prefix_b, suf_b+1)
    # create span a and b

    a_pfx_node, a_suf_node = get_beam_from_past(
        beam_a, prefix_a-1), get_beam_from_past(beam_a, suf_a+1)
    span_a = Span(a_tokens[prefix_a: suf_a+1],
                  a_tokens_str[prefix_a:suf_a+1],
                  score=score_a,
                  prefix=a_tokens[:prefix_a],
                  suffix=a_tokens[suf_a +
                                  1:], prefix_node=a_pfx_node, suffix_node=a_suf_node
                  )
    b_pfx_node, b_suf_node = get_beam_from_past(
        beam_b, prefix_b-1), get_beam_from_past(beam_b, suf_b+1)
    span_b = Span(b_tokens[prefix_b:suf_b+1], b_tokens_str[prefix_b:suf_b+1],
                  score=score_b, prefix=b_tokens[:prefix_b], suffix=b_tokens[suf_b+1:],
                  prefix_node=b_pfx_node, suffix_node=b_suf_node)

    if merge_to_a or score_a > score_b:
        beam_a.add_merge_record(span_a, span_b, beam_b.merge)
        return [beam_a, None]
    else:
        beam_b.add_merge_record(span_b, span_a, beam_a.merge)
        return [None, beam_b]


def async_recomb(candidates: List[BeamState], gen_hash: GenHash, ngram_suffix, len_diff):  # TODO
    output = []
    for cand in candidates:
        if cand in output:
            continue
        tokens = cand.token_full
        # if tokens[-3:] == [81, 63, 92]:
        #     print()

        possible_matches = gen_hash.query(tokens)
        if len(possible_matches) <= 1:
            output.append(cand)
            continue
        possible_matches = [x for x in possible_matches if x.uid != cand.uid]
        tmp_cand = cand
        for match in possible_matches:
            out_a, out_b = merge_compare(
                tmp_cand, match, ngram_suffix=ngram_suffix, len_diff=len_diff)
            if out_a == None or out_b == None:
                tmp_cand = out_b or out_a
        output.append(tmp_cand)
    # dedup
    num_prior_merge = len(output)
    for idx in range(len(output)):
        oid = output[idx].uid
        dups = [jdx for jdx in range(
            len(output)) if output[jdx] and output[jdx].uid == oid]
        if len(dups) > 1:
            output[idx] = None
    output = [x for x in output if x]
    logging.info(f"Merge from {num_prior_merge} to {len(output)}")
    return output


def render_name(doc_input_ids, beam_sz, max_len, ngram_suffix, len_diff):
    first_few_tokens = doc_input_ids.tolist()[0][1:10]
    txt = tokenizer.decode(first_few_tokens)
    params = [beam_sz, max_len, ngram_suffix, len_diff]
    params = "_".join([str(x) for x in params])
    return txt+params


def recomb_beam_search(doc_input_ids, model,  pad_token_id=0, eos_token_id=21, beam_sz=5, max_len=20, num_return_hypo=10000, ngram_suffix=4, len_diff=5):
    gen_hash = GenHash(ngram=ngram_suffix)

    whole_beam = [
        BeamState(cur_idx_in_distb=0,
                  prob_distrib=[1., 0, 0, 0, 0],
                  token_id_distb=[eos_token_id] + [pad_token_id]*4)]
    for t in range(max_len):
        candidates = []

        # optimize: first filter out finished nodes, then run all of them together
        # and then run and  cache their simulation
        finished = [beam_item for beam_item in whole_beam if beam_item.finished]
        active = [beam_item for beam_item in whole_beam if not beam_item.finished]

        # get tokens
        # tokens = [x.token_full for x in active]
        for beam_item in active:
            if not debug:
                # prefix
                decoder_input_ids = beam_item.get_tokens_as_input()
                output_tokens, output_prob, output_score, _ = run_inference_step(
                    model, doc_input_ids, decoder_input_ids=decoder_input_ids, device=device, output_dec_hid=False, T=1)

                # pred_entropy = entropy(output_prob.cpu().numpy(), axis=-1)[0]
                # print(pnum(pred_entropy))
                # dynamic_k = min(BS, t+1)
                dynamic_k = beam_sz
                values, indices = torch.topk(output_prob, k=dynamic_k)
                values = values[0].tolist()
                indices = indices[0].tolist()
                # trim
                # values = [x for x in values if x > 0.01]
                # indices = indices[:len(values)]
            else:
                values, indices = fake_model_output()      # replace it with something real
                values = values.tolist()
                indices = indices.tolist()

            for idx, v, i in zip(range(beam_sz), values, indices):
                tmp_state = BeamState(idx, values, indices, prev=beam_item)
                gen_hash.add(beam_item.token_full + [indices[idx]], tmp_state)
                candidates.append(tmp_state)

        # sort candidates by scores; these are active candidates of the current step
        sorted_candidates = sorted(
            candidates, key=lambda x: x.get_score_sum(), reverse=True)

        # set max possible hypo
        active_candidates = sorted_candidates[:num_return_hypo]
        len_before_recombination = len(active_candidates)
        whole_beam = active_candidates + finished
        whole_beam = async_recomb(whole_beam, gen_hash, ngram_suffix, len_diff)
        # whole_beam = recombined + finished
        # for b in whole_beam:
        #     print(b)
        # print('-')

    logging.info(f"#Whole Beam: {len(whole_beam)}, #finished: {len(finished)}")
    outputs = []
    for unit in finished:
        logging.info(repr(unit))
        outputs.append(pprint(unit.token_full))

    fname = render_name(doc_input_ids, beam_sz, max_len,
                        ngram_suffix, len_diff) + '.pkl'
    with open(f"vizs/{fname}", 'wb') as fd:
        pickle.dump(whole_beam, fd)

    # score = eval_group_diversity(outputs)
    return outputs


def run_example(document):
    doc_input_ids = tokenizer(document, return_tensors='pt')[
        'input_ids'][:, :800]
    doc_input_ids = doc_input_ids.to(device)
    recomb_diverse_score = recomb_beam_search(
        doc_input_ids, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, )
    # logger.info(f"Diverse Score: {recomb_diverse_score}")
    return recomb_diverse_score


if __name__ == '__main__':

    # logging.info(args)
    # args.device = device
    setup_logger(name=f"example")
    nexample = 20
    cnt = 0
    all_scores = []
    all_scores_beam = []
    all_scores_greey = []
    for example in dataset:
        cnt += 1
        document = example['document']
        sents = document.split('\n')
        inp = "\n".join(sents[:10])[:2000]

        doc_id = example['id']
        ref_sum = example['summary']
        logging.info(f"\n\n===Inp Doc: {document[:2000]}\n---Sum: {ref_sum}")
        score = run_example(
            inp)
        all_scores.append(score)
        if cnt > nexample:
            break
    logging.info(f"Ref: {statistics.mean(all_scores)}")
