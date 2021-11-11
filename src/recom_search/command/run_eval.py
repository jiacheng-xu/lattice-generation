import pickle
from src.recom_search.model.model_output import SearchModelOutput
from src.recom_search.model.model_astar import a_star

from src.recom_search.evaluation.eval_bench import rouge_single_pair
import pandas as pd
from collections import defaultdict
from src.recom_search.model.model_explore_then_gen import explore_then_gen
from src.recom_search.model.model_bfs import best_first_search
from src.recom_search.model.baseline import baseline_recomb_sample, recomb_baseline
from src.recom_search.model.generic_search import GenericSearch
from src.recom_search.evaluation.eval_bench import eval_main, np_overlap, rouge, self_bleu
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    HammingDiversityLogitsProcessor,
    BeamSearchScorer,
)

from src.recom_search.model.util import *


def run_recom_bs(args, model, input_doc, param_sim_function):
    input_ids = tokenizer(
        input_doc, return_tensors="pt").input_ids.to(args.device)
    output = recomb_baseline(doc_input_ids=input_ids, param_sim_function=param_sim_function,  eos_token_id=tokenizer.eos_token_id,
                             model=model, debug=False, beam_size=args.beam_size, max_len=args.max_len, num_return_hypo=args.beam_size)

    mo = SearchModelOutput(ends=output)
    return mo


def run_recom_sample(args, model, input_doc, param_sim_function) -> SearchModelOutput:
    input_ids = tokenizer(
        input_doc, return_tensors="pt").input_ids.to(args.device)
    output = baseline_recomb_sample(doc_input_ids=input_ids, param_sim_function=param_sim_function,  eos_token_id=tokenizer.eos_token_id,
                                    model=model, debug=False, max_len=args.max_len, num_return_hypo=args.beam_size, top_p=args.top_p)

    mo = SearchModelOutput(ends=output)
    return mo


def run_a_star(args, model,tokenizer, inp,dec_prefix, param_sim_function, config_search) -> SearchModelOutput:

    config_heu = {
        'heu_seq_score': args.heu_seq_score,
        'heu_seq_score_len_rwd': args.heu_seq_score_len_rwd,
        'heu_pos': args.heu_pos,
        'heu_ent': args.heu_ent,
        'heu_word': args.heu_word
    }
    input_ids = tokenizer(
        inp, return_tensors="pt").input_ids.to(args.device)
    comp_budget = args.max_len * args.beam_size
    output = a_star(doc_input_ids=input_ids, model=model, tokenizer=tokenizer,param_sim_function=param_sim_function, dec_prefix=dec_prefix,avg_score=args.avg_score,
                    max_len=args.max_len, k_best=5, comp_budget=comp_budget, config_heu=config_heu, config_search=config_search)

    mo = SearchModelOutput(ends=output)
    return mo


def run_explore_then_generate(args, model, inp):
    param_sim_function = {
        'ngram_suffix': args.ngram_suffix,
        'len_diff': args.len_diff
    }
    heu_config = {
        'heu_seq_score': args.heu_seq_score,
        'heu_seq_score_len_rwd': args.heu_seq_score_len_rwd,
        'heu_pos': args.heu_pos,
        'heu_ent': args.heu_ent,
        'heu_word': args.heu_word
    }
    input_ids = tokenizer(
        inp, return_tensors="pt").input_ids.to(args.device)
    num_return_hypo = args.max_len * args.beam_size
    output = explore_then_gen(doc_input_ids=input_ids, model=model, param_sim_function=param_sim_function,
                              eos_token_id=tokenizer.eos_token_id,  max_len=args.max_len, k_best=5, num_return_hypo=num_return_hypo, heu_config=heu_config)

    return output


def run_best(args, model, inp):
    param_sim_function = {
        'ngram_suffix': args.ngram_suffix,
        'len_diff': args.len_diff
    }
    heu_config = {
        'heu_seq_score': args.heu_seq_score,
        'heu_seq_score_len_rwd': args.heu_seq_score_len_rwd,
        'heu_pos': args.heu_pos,
        'heu_ent': args.heu_ent,
        'heu_word': args.heu_word
    }
    input_ids = tokenizer(
        inp, return_tensors="pt").input_ids.to(args.device)
    num_return_hypo = args.max_len * args.beam_size
    output = best_first_search(doc_input_ids=input_ids, model=model, param_sim_function=param_sim_function, eos_token_id=tokenizer.eos_token_id,
                               explore_steps=args.extra_steps, max_len=args.max_len, k_best=5, num_return_hypo=num_return_hypo, heu_config=heu_config, min_path=args.min_path)

    return output


def run_baseline(args, model, inp):
    if args.model == 'greedy':
        gs = GenericSearch(model, tokenizer,
                           device=args.device, beam_size=1, do_sample=False, min_len=args.min_len, max_len=args.max_len, num_beam_hyps_to_keep=1)
    elif args.model == 'bs':
        gs = GenericSearch(model, tokenizer,
                           device=args.device, beam_size=args.beam_size, do_sample=False,
                           min_len=args.min_len,
                           max_len=args.max_len,
                           num_beam_hyps_to_keep=args.beam_size
                           )
    elif args.model == 'dbs':
        gs = GenericSearch(model, tokenizer,
                           device=args.device, beam_size=args.beam_size, do_sample=False,
                           min_len=args.min_len, max_len=args.max_len,
                           num_beam_groups=args.beam_group,
                           diversity_penalty=args.hamming_penalty,
                           num_beam_hyps_to_keep=args.beam_size
                           )
    elif args.model == 'topp':
        gs = GenericSearch(model, tokenizer,
                           device=args.device, beam_size=1, do_sample=True, min_len=args.min_len, max_len=args.max_len, num_beam_hyps_to_keep=args.beam_size,
                           top_p=args.top_p)
    elif args.model == 'temp':
        gs = GenericSearch(model, tokenizer,
                           device=args.device, beam_size=1, do_sample=True,
                           min_len=args.min_len, max_len=args.max_len, num_beam_hyps_to_keep=args.beam_size,
                           temperature=args.temp
                           )
    else:
        raise NotImplementedError
    output_dict = gs.run(inp)

    return output_dict
    # output should be a list of str


def main(args, tokenizer, model, dataset,dec_prefix):

    # logging.info(args)
    nexample = args.nexample
    cnt = 0
    for idx,example in enumerate(dataset):
        cnt += 1
        if args.task.startswith('mt'):
            document = example[0]
            ref_sum = example[1]
            inp = document
            doc_id = idx
        elif args.dataset == 'cnndm':
            document = example['article']
            doc_id = example['id']
            ref_sum = example['highlights']
        else:
            document = example['document']
            sents = document.split('\n')
            inp = "\n".join(sents[:10])[:5000]
            ref_sum = example['summary']
            doc_id = example['id']
        # if 'Apple' not in document:
        #     continue
        
        
        logging.info(f"\n\n===Inp Doc: {document[:2000]}\n---Sum: {ref_sum}")
        param_sim_function = {
            'ngram_suffix': args.ngram_suffix,
            'len_diff': args.len_diff,
            'merge':args.merge
        }
        config_search = {
                'post': args.post,
                'post_ratio': args.post_ratio,  # ratio of model calls left for post finishing
                'adhoc': args.adhoc,
                'heu': args.use_heu
            }
        
        if args.model in ['dbs', 'bs', 'greedy', 'topp', 'temp']:
            output = run_baseline(args, model, inp)
        elif args.model == 'recom_bs':
            output = run_recom_bs(args, model, inp, param_sim_function)
        elif args.model == 'recom_sample':
            output = run_recom_sample(args, model, inp, param_sim_function)
        elif args.model == 'astar':
            output = run_a_star(
                args, model,tokenizer, inp,dec_prefix, param_sim_function, config_search=config_search)
        output.reference = ref_sum
        output.doc_id = doc_id
        output.document = document
        output.args = args
        combined_dict = {**config_search, **param_sim_function}
        combined_dict['avgsco'] = args.avg_score
        combined_dict['lenrwd'] = args.heu_seq_score_len_rwd
        combined_dict['topp'] = args.top_p

        fname = render_name(args.task, args.dataset,args.model, doc_id, document[:10], args.beam_size,args.max_len,combined_dict) + '.pkl'
        with open(f"vizs/{fname}", 'wb') as fd:
            pickle.dump(output, fd)

        # break
        if cnt > nexample:
            break


# from src.recom_search.model.util import tokenizer
from src.recom_search.model.setup import tokenizer, model, dataset, dec_prefix,args

if __name__ == "__main__":
    # execute only if run as a script
    main(args, tokenizer, model, dataset,dec_prefix)
