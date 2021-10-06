from pandas.core.dtypes import dtypes
from src.recom_search.evaluation.eval_bench import rouge_single_pair
import pandas as pd
from collections import defaultdict
from src.recom_search.model.model_explore_then_gen import explore_then_gen
from src.recom_search.model.model_bfs import best_first_search
from src.recom_search.model.baseline import recomb_baseline
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

assert model


def process_arg():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model", type=str, choices=['dbs', 'bs', 'greedy', 'topp', 'temp', 'recom', 'best', 'exp_gen'], default='bs')
    parser.add_argument('-beam_size', type=int, default=10)
    parser.add_argument('-nexample', type=int, default=50)

    parser.add_argument('-top_p', type=float, default=0.8)
    parser.add_argument('-temp', type=float, default=1.5)
    parser.add_argument('-beam_group', type=int, default=4)
    parser.add_argument('-hamming_penalty', type=float, default=0.0)
    parser.add_argument('-extra_steps', type=int, default=10)
    parser.add_argument('-min_len', type=int, default=13)
    parser.add_argument('-max_len', type=int, default=25)
    parser.add_argument('-num_beam_hyps_to_keep', type=int, default=100)
    parser.add_argument('-ngram_suffix', type=int, default=3)
    parser.add_argument('-len_diff', type=int, default=5)
    parser.add_argument('-heu_seq_score', type=float, default=0.0,
                        help='Heuristic: consider the score of previously generated sequence. this is the weight term for that')
    parser.add_argument('-heu_seq_score_len_rwd', type=float,
                        default=0.0, help='Length reward term in heu_seq_score.')
    parser.add_argument('-heu_pos', type=float, default=0.0,
                        help='Heuristic for position bias')
    parser.add_argument('-heu_ent', type=float, default=0.0,
                        help='Heuristic for entropy.')
    parser.add_argument('-heu_word', type=float, default=0.0,
                        help='Heuristic for good token.')
    parser.add_argument('-min_path', type=int, default=0,
                        help='Bool indicator of if min_path or not')

    # parser.add_argument("-beam_ent", type=str2bool, nargs='?', const=True,default=False, help="Use entropy to dynamically operate beam.")
    args = parser.parse_args()
    return args


def run_recom(args, model, input_doc):
    param_sim_function = {
        'ngram_suffix': args.ngram_suffix,
        'len_diff': args.len_diff
    }

    input_ids = tokenizer(
        input_doc, return_tensors="pt").input_ids.to(args.device)
    output = recomb_baseline(doc_input_ids=input_ids, param_sim_function=param_sim_function,  eos_token_id=tokenizer.eos_token_id,
                             model=model, debug=False, beam_size=args.beam_size, max_len=args.max_len, num_return_hypo=args.beam_size)
    # output = recomb_beam_search(input_ids, model, pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,beam_sz=args.beam_size, max_len=args.max_len, num_return_hypo=args.beam_size,ngram_suffix=args.ngram_suffix, len_diff=args.len_diff)

    return output


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
                           device=device, beam_size=1, do_sample=False, min_len=args.min_len, max_len=args.max_len, num_beam_hyps_to_keep=1)
    elif args.model == 'bs':
        gs = GenericSearch(model, tokenizer,
                           device=device, beam_size=args.beam_size, do_sample=False,
                           min_len=args.min_len,
                           max_len=args.max_len,
                           num_beam_hyps_to_keep=args.beam_size
                           )
    elif args.model == 'dbs':
        gs = GenericSearch(model, tokenizer,
                           device=device, beam_size=args.beam_size, do_sample=False,
                           min_len=args.min_len, max_len=args.max_len,
                           num_beam_groups=args.beam_group,
                           diversity_penalty=args.hamming_penalty,
                           num_beam_hyps_to_keep=args.beam_size
                           )
    elif args.model == 'topp':
        gs = GenericSearch(model, tokenizer,
                           device=device, beam_size=1, do_sample=True, min_len=args.min_len, max_len=args.max_len, num_beam_hyps_to_keep=args.beam_size,
                           top_p=args.top_p)
    elif args.model == 'temp':
        gs = GenericSearch(model, tokenizer,
                           device=device, beam_size=1, do_sample=True,
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

    all_outputs = []
    all_summaries = []
    all_branch = []
    all_model_scores = []
    all_rouge_scores = []

    all_top_model_scores = []
    all_low_model_scores = []
    all_top_rouge_scores = []
    for example in dataset:
        cnt += 1
        document = example['document']
        sents = document.split('\n')
        inp = "\n".join(sents[:10])[:5000]

        doc_id = example['id']
        ref_sum = example['summary']
        logging.info(f"\n\n===Inp Doc: {document[:2000]}\n---Sum: {ref_sum}")
        if args.model in ['dbs', 'bs', 'greedy', 'topp', 'temp']:
            output_dict = run_baseline(args, model, inp)
        elif args.model == 'recom':
            output = run_recom(args, model, inp)
        elif args.model == 'best':
            output = run_best(args, model, inp)
        elif args.model == 'exp_gen':
            output = run_explore_then_generate(args=args, model=model, inp=inp)
            # for k, v in stat.items():
            #     d[k].append(v)
        

        scores = output_dict['score']
        output = output_dict['output']
        branch = output_dict['branch']
        n_outputs = len(output)
        if scores:
            assert n_outputs == len(scores)
        all_outputs += output
        all_summaries += [ref_sum]*n_outputs
        all_model_scores += scores
        all_branch += [branch] * n_outputs
        rouge_scores = [rouge_single_pair(
            x, y) for x, y in zip(output, [ref_sum]*n_outputs)]
        all_rouge_scores += rouge_scores

        # extract ROUGE of highest score, and score of highest ROUGE
        index_of_highest_score = np.argsort(scores)[-1]
        highest_score = scores[index_of_highest_score]
        all_top_model_scores += [highest_score] * n_outputs

        index_of_lowest_score = np.argsort(scores)[0]
        lowest_score = scores[index_of_lowest_score]
        all_low_model_scores += [lowest_score] * n_outputs

        index_of_highest_rouge = np.argsort(rouge_scores)[-1]
        highest_rouge = rouge_scores[index_of_highest_rouge]
        all_top_rouge_scores += [highest_rouge] * n_outputs

        # break
        if cnt > nexample:
            break

    # construct panda data frame
    d = {
        "gen": all_outputs,
        "ref": all_summaries,
        "quant": all_branch,
        'score': pd.to_numeric(all_model_scores),
        'rouge': pd.to_numeric(all_rouge_scores),
        'top_score': pd.to_numeric(all_top_model_scores),
        'low_score': pd.to_numeric(all_low_model_scores),
        'top_rouge': pd.to_numeric(all_top_rouge_scores)
    }
    df = pd.DataFrame(d)
    import pickle
    with open('tmp_data.pkl', 'wb') as fd:
        pickle.dump(df, fd)

    # for summ, out in zip(all_summaries, all_outputs):
    #     output_d = eval_main(out, summ)
    #     for k, v in output_d.items():
    #         d[k].append(v)
    # nums_brief = []
    # stat_result = analyze_stat_dict(d)
    # logging.info(f"STAT: {stat_result}")
    # for k, v in d.items():
    #     avg = statistics.mean(v)
    #     logging.info(f"{k}:{pnum(avg) }")
    #     nums_brief.append(pnum(avg))

    # logging.info(",".join(nums_brief))
    # # viz in one line
    # viz = [args.model, args.hamming_penalty, args.top_p, args.temp, args.extra_steps] + \
    #     nums_brief + list(stat_result.values())
    # viz = [str(x) for x in viz]
    # logging.info(','.join(viz))


if __name__ == "__main__":
    # execute only if run as a script
    args = process_arg()
    args.device = device
    setup_logger(name=f"{args.model}")
    tokenizer, model, dataset = setup_model()
    main(args, tokenizer, model, dataset)
