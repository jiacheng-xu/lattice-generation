from collections import defaultdict
from src.recom_search.model.model_bfs import best_first_search
from src.recom_search.model.baseline import recomb_baseline
from src.recom_search.model.generic_search import GenericSearch
from src.recom_search.evaluation.eval_bench import eval_main, np_overlap, rouge, self_bleu


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
        "-model", type=str, choices=['dbs', 'bs', 'greedy', 'topp', 'temp', 'recom', 'best'], default='best')
    parser.add_argument('-beam_size', type=int, default=10)
    parser.add_argument('-nexample', type=int, default=20)

    parser.add_argument('-top_p', type=float, default=0.8)
    parser.add_argument('-temp', type=float, default=1.5)
    parser.add_argument('-beam_group', type=int, default=4)
    parser.add_argument('-hamming_penalty', type=float, default=0.0)
    parser.add_argument('-extra_steps', type=int, default=10)
    parser.add_argument('-min_len', type=int, default=10)
    parser.add_argument('-max_len', type=int, default=25)
    parser.add_argument('-num_beam_hyps_to_keep', type=int, default=100)
    parser.add_argument('-ngram_suffix', type=int, default=3)
    parser.add_argument('-len_diff', type=int, default=5)
    parser.add_argument('-heuristic_position_bias', type=float, default=0.0,
                        help='Add more score to the begining of a sentence.')
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


def run_best(args, model, inp):
    param_sim_function = {
        'ngram_suffix': args.ngram_suffix,
        'len_diff': args.len_diff
    }
    input_ids = tokenizer(
        inp, return_tensors="pt").input_ids.to(args.device)
    num_return_hypo = args.max_len * args.beam_size
    output = best_first_search(doc_input_ids=input_ids, model=model, param_sim_function=param_sim_function, eos_token_id=tokenizer.eos_token_id,
                               explore_steps=args.extra_steps, max_len=args.max_len, k_best=5, num_return_hypo=num_return_hypo, position_bias=args.heuristic_position_bias)

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
    output = gs.run(inp)

    return output
    # output should be a list of str


def main(args, tokenizer, model, dataset):

    # logging.info(args)
    nexample = args.nexample
    cnt = 0
    all_outputs = []
    all_summaries = []
    d = defaultdict(list)

    for example in dataset:
        cnt += 1
        document = example['document']
        sents = document.split('\n')
        inp = "\n".join(sents[:10])[:5000]

        doc_id = example['id']
        ref_sum = example['summary']
        logging.info(f"\n\n===Inp Doc: {document[:2000]}\n---Sum: {ref_sum}")
        if args.model in ['dbs', 'bs', 'greedy', 'topp', 'temp']:
            output = run_baseline(args, model, inp)
        elif args.model == 'recom':
            output = run_recom(args, model, inp)
        elif args.model == 'best':
            output = run_best(args, model, inp)
            # for k, v in stat.items():
            #     d[k].append(v)
        all_outputs.append(output)
        all_summaries.append(ref_sum)
        if cnt > nexample:
            break

    for summ, out in zip(all_summaries, all_outputs):
        output_d = eval_main(out, summ)
        for k, v in output_d.items():
            d[k].append(v)
    nums_brief = []
    stat_result = analyze_stat_dict(d)
    logging.info(f"STAT: {stat_result}")
    for k, v in d.items():
        avg = statistics.mean(v)
        logging.info(f"{k}:{pnum(avg) }")
        nums_brief.append(pnum(avg))

    logging.info(",".join(nums_brief))
    # viz in one line
    viz = [args.model, args.hamming_penalty, args.top_p, args.temp, args.extra_steps] + \
        nums_brief + list(stat_result.values())
    viz = [str(x) for x in viz]
    logging.info(','.join(viz))


if __name__ == "__main__":
    # execute only if run as a script
    args = process_arg()
    args.device = device
    setup_logger(name=f"{args.model}")
    tokenizer, model, dataset = setup_model()
    main(args, tokenizer, model, dataset)
