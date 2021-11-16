
from rouge_score import rouge_scorer
import statistics
from collections import defaultdict
from collections import Counter
import imp
from typing import List
import nltk
import spacy
nlp = spacy.load("en_core_web_sm")

full_rouge_scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


from sacrebleu.metrics import BLEU
bleu_scorer = BLEU(effective_order=True)

def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def tokenize_sentences(inp_group):
    output = []
    for inp in inp_group:
        doc = nlp(inp)
        output.append([token.text for token in doc])
    return output


def extract_np(inp_group):
    output = []
    for inp in inp_group:
        doc = nlp(inp)
        output.append(set([chunk.root.text for chunk in doc.noun_chunks]))
    return output


def np_overlap(inp_group: List[str]):
    tok_inputs = extract_np(inp_group)
    scores = []
    for idx, inp in enumerate(tok_inputs):
        for jdx, x in enumerate(tok_inputs):
            if jdx >= idx:
                continue
            rate = len(inp.intersection(x)) / len(inp.union(x))
            scores.append(rate)

    return statistics.mean(scores)


def self_bleu(inp_group: List[str]):
    # tok_inputs = tokenize_sentences(inp_group)
    bleu_scores = []
    for idx, inp in enumerate(inp_group):
        # bleu_score = nltk.translate.bleu_score.sentence_bleu([x for jdx, x in enumerate(tok_inputs) if jdx != idx], inp)
        bleu_score = bleu_scorer.sentence_score(inp, [x for jdx, x in enumerate(inp_group) if jdx != idx])
        bleu_scores.append(bleu_score.score)
    return statistics.mean(bleu_scores)


def repetition(inp_group: List[str], threshold=3):
    tok_inputs = tokenize_sentences(inp_group)
    cnt = Counter()
    all_ngrams = [_get_ngrams(3, tok_sent) for tok_sent in tok_inputs]
    [[cnt.update(element) for element in ngrams] for ngrams in all_ngrams]
    total_len = len(cnt)
    matter = 0
    for k, v in cnt.items():
        if v >= threshold:
            matter += 1
    return matter / total_len


def rouge_single_pair(cand: str, ref: str, metric='rouge1'):
    s = full_rouge_scorer.score(cand, ref)
    return s[metric].fmeasure


def rouge(inp_group, reference: str) -> dict:
    scores = defaultdict(list)
    for inp in inp_group:
        s = full_rouge_scorer.score(inp, reference)
        f1,f2, fl = s['rouge1'].fmeasure, s['rouge2'].fmeasure, s['rougeL'].fmeasure
        scores['r1'].append(f1)
        scores['r2'].append(f2)
        scores['rl'].append(fl)
    d = {}
    for k, v in scores.items():
        avg = statistics.mean(v)
        d[k] = avg
    return d


def eval_main(inp_group, reference, prefix=""):
    dict_rouge = rouge(inp_group, reference)
    if len(inp_group) <= 1:
        d = {
            'REP': 0,
            'SELF_BLEU': 0,
            # 'NP_OVERLAP': 0,
        }
    else:
        d = {
           
            'REP': repetition(inp_group),
            'SELF_BLEU': self_bleu(inp_group),
            # 'NP_OVERLAP': np_overlap(inp_group),
        }
    d = {**d, **dict_rouge}
    #update d with key prefix
    new_d = {}
    for k,v in d.items():
        new_d[f"{prefix}_{k}"] = v
    return new_d
