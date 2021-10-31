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
import argparse

# assert model


def process_arg():

    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=str, default='cuda:2')
    parser.add_argument(
        "-model", type=str, choices=['dbs', 'bs', 'greedy', 'topp', 'temp', 'recom_bs', 'recom_sample',  'astar'], default='bs')
    parser.add_argument('-beam_size', type=int, default=15)
    parser.add_argument('-nexample', type=int, default=20)

    parser.add_argument('-top_p', type=float, default=0.9)
    parser.add_argument('-temp', type=float, default=1.5)
    parser.add_argument('-beam_group', type=int, default=5)
    parser.add_argument('-hamming_penalty', type=float, default=0.0)
    parser.add_argument('-extra_steps', type=int, default=10)
    parser.add_argument('-min_len', type=int, default=13)
    parser.add_argument('-max_len', type=int, default=35)
    parser.add_argument('-num_beam_hyps_to_keep', type=int, default=100)
    parser.add_argument('-ngram_suffix', type=int, default=4)
    parser.add_argument('-len_diff', type=int, default=5)

    parser.add_argument('-avg_score', type=str2bool, nargs='?',
                        const=True, default=False, help='use average model score')

    parser.add_argument('-use_heu', type=str2bool, nargs='?',
                        const=True, default=False, help='our model: do we use heuristic')
    parser.add_argument('-post', type=str2bool, nargs='?',
                        const=True, default=False, help='our model: enforce the model to generate after exploration')
    parser.add_argument('-adhoc', type=str2bool, nargs='?',
                        const=True, default=False, help='our model: always generate till the end once touch a node')
    parser.add_argument('-post_ratio', type=float, default=0.4,
                        help='our model: ratio of resource allocation')

    parser.add_argument('-heu_seq_score', type=float, default=0.0,
                        help='Heuristic: consider the score of previously generated sequence. this is the weight term for that')
    parser.add_argument('-heu_seq_score_len_rwd', type=float,
                        default=0.0, help='Length reward term in heu_seq_score.')
    parser.add_argument('-heu_pos', type=float, default=0.0,
                        help='Heuristic for position bias')
    parser.add_argument('-heu_ent', type=float, default=0.5,
                        help='Heuristic for entropy.')
    parser.add_argument('-heu_word', type=float, default=0.0,
                        help='Heuristic for good token.')

    # parser.add_argument('-min_path', type=int, default=0,help='Bool indicator of if min_path or not')

    # parser.add_argument("-beam_ent", type=str2bool, nargs='?', const=True,default=False, help="Use entropy to dynamically operate beam.")
    args = parser.parse_args()
    return args


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


def run_a_star(args, model,tokenizer, inp, param_sim_function, config_search) -> SearchModelOutput:

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
    output = a_star(doc_input_ids=input_ids, model=model, tokenizer=tokenizer,param_sim_function=param_sim_function, eos_token_id=tokenizer.eos_token_id,avg_score=args.avg_score,
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


def main(args, tokenizer, model, dataset):

    # logging.info(args)
    nexample = args.nexample
    cnt = 0
    for example in dataset:
        cnt += 1
        document = example['document']
        sents = document.split('\n')
        inp = "\n".join(sents[:10])[:5000]
        # if 'Apple' not in document:
        #     continue
        doc_id = example['id']
        ref_sum = example['summary']
        logging.info(f"\n\n===Inp Doc: {document[:2000]}\n---Sum: {ref_sum}")
        param_sim_function = {
            'ngram_suffix': args.ngram_suffix,
            'len_diff': args.len_diff
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
                args, model,tokenizer, inp, param_sim_function, config_search=config_search)
        output.reference = ref_sum
        output.doc_id = doc_id
        output.document = document
        output.args = args
        combined_dict = {**config_search, **param_sim_function}
        combined_dict['avgsco'] = args.avg_score
        combined_dict['lenrwd'] = args.heu_seq_score_len_rwd
        combined_dict['topp'] = args.top_p

        fname = render_name(args.model, doc_id, document[:10], args.beam_size,args.max_len,combined_dict) + '.pkl'
        with open(f"vizs/{fname}", 'wb') as fd:
            pickle.dump(output, fd)

        # break
        if cnt > nexample:
            break



if __name__ == "__main__":
    # execute only if run as a script
    args = process_arg()

    setup_logger(name=f"{args.model}")
    tokenizer, model, dataset = setup_model(args.device)
    main(args, tokenizer, model, dataset)
