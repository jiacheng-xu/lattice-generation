from re import split
import statistics
from util import str2bool
import argparse
from typing import List
import math
from scipy.stats import entropy
from numpy.lib.utils import who
from util import *
from recomb_data_struct import BeamState
from recombination_prototype import eval_group_diversity

LEN_DIFF = 4  # max diff of two string
MAX_STEP = 30  # max gen steps
BS = 20
NGRAM_SUF = 2  # last NGRAM_SUF tokens match
RWD_LEN = 0.08  # score = \sum RWD_len + log p(y)
logging.info(f"BS:{BS} SUFFIX:{NGRAM_SUF} MAX_STEP:{MAX_STEP}")

debug = False

model_name = 'sshleifer/distilbart-xsum-12-6'
tokenizer = BartTokenizer.from_pretrained(model_name)


def sublist(lst1, lst2):
    return set(lst1) <= set(lst2)


def fake_model_output(vocab_size=20, k=BS):
    output = torch.rand(vocab_size) * 20
    softmax_scores = torch.nn.functional.softmax(output)
    return torch.topk(softmax_scores, k=k)


class Span():
    def __init__(self, tokens, token_strs, score: float, prefix=[], suffix=[]) -> None:
        # [A: [token, token, token], [score, score, score], B: [token, token], [score, score], .....]
        # prefix if provided
        # we might want something like model repr in future
        self.tokens = tokens
        self.score = score
        self.prefix = prefix
        self.suffix = suffix
        self.token_strs = token_strs

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


def merge_compare(beam_a, beam_b):
    # we assume we are matching the suffix of a and b although their length can be different
    # try to merge a -> b
    # Step 1: suffix match
    a_tokens = beam_a.get_tokens()
    b_tokens = beam_b.get_tokens()

    if len(a_tokens) > NGRAM_SUF and len(b_tokens) > NGRAM_SUF:
        if a_tokens[-NGRAM_SUF:] == b_tokens[-NGRAM_SUF:]:
            logging.debug(f"Stage 1: Suffix match SUCCESS")
        else:
            return [beam_a, beam_b]
    else:
        return [beam_a, beam_b]

    # Stage 2: length
    if abs(len(a_tokens) - len(b_tokens)) < LEN_DIFF:
        logging.debug(f"Stage 2: Len Diff SUCCESS")
    else:
        logging.debug(f"Stage 2: Len Diff FAIL")
        return [beam_a, beam_b]
    a_tokens_str = beam_a.get_tokens_str()
    b_tokens_str = beam_b.get_tokens_str()
    # Stage 3: let's merge!
    if a_tokens == b_tokens:
        score_a = beam_a.get_score_sum()
        score_b = beam_b.get_score_sum()
        span_a = Span(a_tokens, a_tokens_str, score=score_a,
                      prefix=[], suffix=a_tokens)
        span_b = Span(b_tokens, b_tokens_str, score=score_b,
                      prefix=[], suffix=b_tokens)
        if score_a > score_b:
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
    span_a = Span(a_tokens[prefix_a: suf_a+1], a_tokens_str[prefix_a:suf_a+1],
                  score=score_a, prefix=a_tokens[:prefix_a], suffix=a_tokens[suf_a+1:])

    span_b = Span(b_tokens[prefix_b:suf_b+1], b_tokens_str[prefix_b:suf_b+1],
                  score=score_b, prefix=b_tokens[:prefix_b], suffix=b_tokens[suf_b+1:])
    if score_a > score_b:
        beam_a.add_merge_record(span_a, span_b, beam_b.merge)
        return [beam_a, None]
    else:
        beam_b.add_merge_record(span_b, span_a, beam_a.merge)
        return [None, beam_b]


def entrance_merge(beam: List[BeamState]):
    for idx, b in enumerate(beam):
        for jdx in range(len(beam)):
            if idx == jdx:
                continue
            cand_a, cand_b = beam[idx], beam[jdx]
            if (not cand_a) or (not cand_b):
                continue    # if has been merged, will be None
            beam[idx], beam[jdx] = merge_compare(cand_a, cand_b)
    beam = [x for x in beam if x != None]
    return beam


def recomb_beam_search(doc_input_ids, pad_token_id=0, eos_token_id=21):
    logging.info(f"\nBEAM SEARCH\n")

    whole_beam = [BeamState(cur_idx_in_distb=0, prob_distrib=[1., 0, 0, 0, 0], token_id_distb=[
                            eos_token_id, pad_token_id, pad_token_id, pad_token_id, pad_token_id])]
    for t in range(MAX_STEP):
        candidates = []
        for beam_item in whole_beam:
            if beam_item.finished:
                candidates.append(beam_item)
                continue

            if not debug:
                # prefix
                decoder_input_ids = beam_item.get_tokens_as_input()
                output_tokens, output_prob, output_score, _ = run_full_model_slim(
                    model, doc_input_ids, decoder_input_ids=decoder_input_ids, device=device, output_dec_hid=False, T=1)

                # pred_entropy = entropy(output_prob.cpu().numpy(), axis=-1)[0]
                # print(pnum(pred_entropy))
                dynamic_k = min(BS, t+1)
                # dynamic_k= BS
                values, indices = torch.topk(output_prob, k=dynamic_k)
                values = values[0].tolist()
                indices = indices[0].tolist()
            else:
                values, indices = fake_model_output()      # replace it with something real
                values = values.tolist()
                indices = indices.tolist()

            for idx, v, i in zip(range(BS), values, indices):
                tmp_state = BeamState(idx, values, indices, prev=beam_item)
                candidates.append(tmp_state)

        # sort candidates by scores
        sorted_candidates = sorted(
            candidates, key=lambda x: x.get_score_sum(), reverse=True)
        whole_beam = sorted_candidates[:BS]
        original_len = len(whole_beam)
        whole_beam = entrance_merge(whole_beam)
        logging.info(f"COUNT: {original_len}->{len(whole_beam)}")

    outputs = []
    for unit in whole_beam:
        logging.info(repr(unit))
        outputs.append(unit.get_output_str())
    score = eval_group_diversity(outputs)

    # for unit in whole_beam:
    #     logging.info(repr(unit))
        # logging.info(unit.get_simple_repr())
        # logging.info(unit.get_complete_repr())

    # scores = []
    # for unit in whole_beam:
    #     score = unit.get_score_sum()
    #     scores.append(score)
    return score


def run_example(document):
    doc_input_ids = tokenizer(document, return_tensors='pt')[
        'input_ids'][:, :800]
    doc_input_ids = doc_input_ids.to(device)
    recomb_diverse_score = recomb_beam_search(doc_input_ids,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    logger.info(f"Diverse Score: {recomb_diverse_score}")
    return recomb_diverse_score

if __name__ == '__main__':

    # logging.info(args)
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
