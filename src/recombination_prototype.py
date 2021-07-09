from re import split
import statistics
from util import str2bool
import argparse
from typing import List
import math
from scipy.stats import entropy
from numpy.lib.utils import who
from util import *

# a beam search algorithm with merge.

L = 8  # max diff of two substrings
MAX_STEP = 30  # max gen steps
BS = 10
SUFFIX = 2  # last two tokens match
logging.info(f"BS:{BS} SUFFIX:{SUFFIX} MAX_STEP:{MAX_STEP}")
merge = False
# Each state: UID, log score, previous tokens, top-K and prob for all prev steps

# When to MERGE? When two generation shares some father, and the suffix equals, and the diff is small (len < L)
# global GLOBAL_UID_CNT
GLOBAL_UID_CNT = 0


def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def eval_group_diversity(group: List[str]):
    group = [_get_ngrams(2, x.split(" ")) for x in group]
    overlap_ratios = []
    for idx in range(len(group)):
        for jdx in range(len(group)):
            if idx != jdx:
                a, b = group[idx], group[jdx]
                overlap = len(a.intersection(b)) / len(a.union(b))
                overlap_ratios.append(overlap)
    return statistics.mean(overlap_ratios)


class Span():
    def __init__(self) -> None:
        # [A: [token, token, token], [score, score, score], B: [token, token], [score, score], .....]
        # prefix if provided
        # we might want something like model repr in future
        pass


def compare_ancestor_of_states(bs1, bs2):
    father1 = bs1.get_ancestor_uid()
    father2 = bs2.get_ancestor_uid()
    # trim
    father1 = father1[:L]
    father2 = father2[:L]
    distance1, distance2 = -1, -1
    for idx, element in enumerate(father1):
        if element in father2:
            distance1 = idx
            break
    if distance1 > 0:
        distance2 = father2.index(element)
    return distance1, distance2

def sublist(lst1, lst2):
    return set(lst1) <= set(lst2)


def fake_model_output(vocab_size=20):
    output = torch.rand(vocab_size) * 20
    softmax_scores = torch.nn.functional.softmax(output)
    return torch.topk(softmax_scores, k=BS)

# we only do exact suffix matching for now


def merge_compare(beam_a, beam_b):
    # Step 1: suffix match
    a_tokens = beam_a.get_tokens()
    b_tokens = beam_b.get_tokens()
    if len(a_tokens) > SUFFIX and len(b_tokens) > SUFFIX:
        target_tokens = a_tokens[:SUFFIX]
        contain = sublist(target_tokens, b_tokens)
        if a_tokens[:SUFFIX] == b_tokens[:SUFFIX]:
            logging.info(f"Stage 1: SUCCESS")
            pass
        else:
            return None
    else:
        return None


def entrance_merge(beam: List[BeamState]):
    for idx, b in enumerate(beam):
        for jdx in range(idx+1, len(beam)):
            cand_a, cand_b = beam[idx], beam[jdx]
            if (not cand_a) or (not cand_b):
                continue
            merge_compare(cand_a, cand_b)


def measure_ref_score(doc_input_ids, ref_sum: str):
    start = tokenizer.eos_token_id
    ref_sum_token_ids = tokenizer(
        [tokenizer.eos_token + ref_sum], return_tensors='pt')['input_ids'][:, 1:-1].to(device)
    all_values = []
    _, timesteps = ref_sum_token_ids.size()
    for _t in range(1, timesteps):
        output_tokens, output_prob, output_score, _ = run_full_model_slim(
            model, doc_input_ids, decoder_input_ids=ref_sum_token_ids[:, :_t], device=device, output_dec_hid=False, T=1)
        # values, indices = torch.topk(output_prob, k=5)
        # values = values[0].tolist()
        # indices = indices[0].tolist()
        all_values.append(math.log(output_prob[0][ref_sum_token_ids[:,_t]].float() ))

    calibrate_all_values = all_values[:MAX_STEP]
    avg_values = statistics.mean(all_values)
    logging.info(f"Ref Score: {pnum(avg_values)}")
    logging.info(
        f"Length Calib Ref Score: {pnum(statistics.mean(calibrate_all_values))}")
    return avg_values

def greedy(doc_input_ids):
    start = tokenizer.eos_token_id
    decoder_inputs = [[tokenizer.eos_token_id]]
    decoder_input_ids = torch.LongTensor(decoder_inputs).to(device)

    all_values = []
    for _t in range(1, MAX_STEP):
        output_tokens, output_prob, output_score, _ = run_full_model_slim(
            model, doc_input_ids, decoder_input_ids=decoder_input_ids, device=device, output_dec_hid=False, T=1)
        values, indices = torch.topk(output_prob, k=5)
        values = values[0].tolist()
        indices = indices[0].tolist()
        if indices[0] == tokenizer.eos_token_id:
            break
        decoder_inputs[0].append(indices[0])
        decoder_input_ids = torch.LongTensor(decoder_inputs).to(device)
        all_values.append(math.log(values[0]))

    calibrate_all_values = all_values[:MAX_STEP]
    avg_values = statistics.mean(all_values)
    logging.info(f"Greedy Output: {tokenizer.decode(decoder_inputs[0])}")
    logging.info(f"Greedy Score: {pnum(avg_values)}")
    logging.info(
        f"Length Calib Greedy Score: {pnum(statistics.mean(calibrate_all_values))}")
    return avg_values


def beam_search(doc_input_ids, greedy_point=2):
    logging.info(f"\nBEAM SEARCH\n")

    whole_beam = [BeamState(cur_idx_in_distb=0, prob_distrib=[1., 0, 0, 0, 0], token_id_distb=[
                            tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.pad_token_id, tokenizer.pad_token_id, tokenizer.pad_token_id])]
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
                # dynamic_k = min(BS, math.ceil(pred_entropy))
                if t >= greedy_point:
                    dynamic_k = 1
                else:
                    dynamic_k = BS
                values, indices = torch.topk(output_prob, k=dynamic_k)
            else:
                values, indices = fake_model_output()      # replace it with something real
            values = values[0].tolist()
            indices = indices[0].tolist()

            for idx, v, i in zip(range(BS), values, indices):
                tmp_state = BeamState(idx, values, indices, prev=beam_item)
                candidates.append(tmp_state)

        # sort candidates by scores
        sorted_candidates = sorted(
            candidates, key=lambda x: x.get_score(), reverse=True)
        whole_beam = sorted_candidates[:BS]

    for unit in whole_beam:
        logging.info(repr(unit))

    for unit in whole_beam:
        logging.info(unit.get_simple_repr())
        # logging.info(unit.get_complete_repr())

    scores = []
    for unit in whole_beam:
        score = unit.get_score()
        scores.append(score)
    return scores

def select_top_p(prob_distribution, target_index=None):
    pass


def run_example(document, ref_sum):
    doc_input_ids = tokenizer(document, return_tensors='pt')[
        'input_ids'][:, :800]
    doc_input_ids = doc_input_ids.to(device)

    ref_score = measure_ref_score(doc_input_ids,ref_sum)
    beam_scores = beam_search(doc_input_ids)
    greedy_score = greedy(doc_input_ids)
    logging.info('done')
    return ref_score, beam_scores, greedy_score


def process_arg():

    parser = argparse.ArgumentParser()
    parser.add_argument("-beam_ent_multiplier", type=float,
                        help="the multiplier of beam entropy")

    parser.add_argument("-beam_ent", type=str2bool, nargs='?', const=True,
                        default=False, help="Use entropy to dynamically operate beam.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = process_arg()
    logging.info(args)
    nexample = 20
    cnt = 0
    all_scores_ref = []
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
        ref_score, beam_scores, greedy_score = run_example(inp, ref_sum.strip())
        all_scores_ref.append(ref_score)
        all_scores_beam.append(statistics.mean(beam_scores))
        all_scores_greey.append(greedy_score)
        if cnt > nexample:
            break
    logging.info(f"Ref: {statistics.mean(all_scores_ref)}")
    logging.info(f"Beam: {statistics.mean(all_scores_beam)}")
    logging.info(f"Greedy: {statistics.mean(all_scores_greey)}")
