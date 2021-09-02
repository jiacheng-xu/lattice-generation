from collections import defaultdict
from src.recom_search.model.generic_search import GenericSearch
from src.recom_search.evaluation.eval_bench import eval_main, np_overlap, rouge, self_bleu

from src.recom_search.model.best_first_search import *
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    HammingDiversityLogitsProcessor,
    BeamSearchScorer,
)

from src.recom_search.model.recomb_proto import *
from src.recom_search.model.util import *
import argparse


def process_arg():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model", type=str, choices=['dbs', 'bs', 'greedy', 'topp', 'temp', 'recom', 'best'], default='best')
    parser.add_argument('-beam_size', type=int, default=20)
    parser.add_argument('-nexample', type=int, default=50)
    
    parser.add_argument('-top_p', type=float, default=0.8)
    parser.add_argument('-temp', type=float, default=1.5)
    parser.add_argument('-beam_group', type=int, default=4)
    parser.add_argument('-hamming_penalty', type=float, default=0.0)
    parser.add_argument('-min_len', type=int, default=10)
    parser.add_argument('-max_len', type=int, default=30)
    parser.add_argument('-num_beam_hyps_to_keep', type=int, default=100)
    # parser.add_argument("-beam_ent", type=str2bool, nargs='?', const=True,default=False, help="Use entropy to dynamically operate beam.")
    args = parser.parse_args()
    return args


def run_recom(args, model, input_doc):
    input_ids = tokenizer(
        input_doc, return_tensors="pt").input_ids.to(args.device)
    output = recomb_beam_search(input_ids, model, pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                beam_sz=args.beam_size, max_len=args.max_len, num_return_hypo=args.beam_size)
    return output


def run_best(args, model, inp):
    input_ids = tokenizer(
        inp, return_tensors="pt").input_ids.to(args.device)
    output = best_first_search(input_ids, model, pad_token_id=tokenizer.pad_token_id,
                               eos_token_id=tokenizer.eos_token_id,  max_len=args.max_len,explore_cnt=args.beam_size)
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
                           device=device, beam_size=args.beam_size, do_sample=False, min_len=args.min_len, max_len=args.max_len,  num_beam_groups=args.beam_group, diversity_penalty=args.hamming_penalty,
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


def main(args):

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
        all_outputs.append(output)
        all_summaries.append(ref_sum)
        if cnt > nexample:
            break

    for summ, out in zip(all_summaries, all_outputs):
        output_d = eval_main(out, summ)
        for k, v in output_d.items():
            d[k].append(v)
    nums_brief = []
    for k, v in d.items():
        avg = statistics.mean(v)
        logging.info(f"{k}:{pnum(avg) }")
        nums_brief.append(pnum(avg))
    logging.info(",".join(nums_brief))


if __name__ == "__main__":
    # execute only if run as a script
    args = process_arg()
    args.device = device
    setup_logger(name=f"{args.model}")
    main(args)
