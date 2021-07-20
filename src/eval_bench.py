from recombination_prototype import _get_ngrams
import statistics
from collections import defaultdict
from collections import Counter
import imp
from typing import List
import nltk
import spacy


nlp = spacy.load("en_core_web_sm")


def tokenize_sentences(inp_group):
    output = []
    for inp in inp_group:
        doc = nlp(inp)
        output.append([token.text for token in doc])
    return output


def self_bleu(inp_group: List[str]):
    tok_inputs = tokenize_sentences(inp_group)
    bleu_scores = []
    for idx, inp in enumerate(tok_inputs):
        bleu_score = nltk.translate.bleu_score.sentence_bleu(
            [x for jdx, x in enumerate(tok_inputs) if jdx != idx], inp)
        bleu_scores.append(bleu_score)
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
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


def rouge(inp_group, reference:str)->float:
    scores = []
    for inp in inp_group:
        s = scorer.score(inp ,reference)
        f1, fl = s['rouge1'].fmeasure, s['rougeL'].fmeasure
        scores.append(f1)
    return statistics.mean(scores)
    

    pass