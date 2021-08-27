from early_stop import best_first_search
from eval_bench import *
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    HammingDiversityLogitsProcessor,
    BeamSearchScorer,
)
from recomb_proto import recomb_beam_search
from util import *


def process_arg():

    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, choices=['dbs', 'bs', 'greedy', 'delay', 'topp', 'recom',' best'], default='best')
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-top_p', type=float, default=0.8)

    parser.add_argument('-beam_group', type=int, default=5)
    parser.add_argument('-hamming_penalty', type=float, default=1.)
    parser.add_argument('-min_len', type=int, default=5)
    parser.add_argument('-max_len', type=int, default=5)
    parser.add_argument('-num_beam_hyps_to_keep', type=int, default=100)
    # parser.add_argument("-beam_ent", type=str2bool, nargs='?', const=True,default=False, help="Use entropy to dynamically operate beam.")
    args = parser.parse_args()
    return args


def run_dbs(args, model, input_doc: str):
    encoder_input_ids = tokenizer(
        input_doc, return_tensors="pt").input_ids.to(args.device)

    # lets run diverse beam search using 6 beams
    num_beams = args.beam_size
    ngroups = args.beam_group
    # define decoder start token ids
    input_ids = torch.ones(
        (num_beams, 1), device=model.device, dtype=torch.long)
    input_ids = input_ids * model.config.decoder_start_token_id
    model_kwargs = {
        "encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)}
    beam_scorer = BeamSearchScorer(
        batch_size=1,
        max_length=args.max_len,
        num_beams=args.beam_size,
        device=model.device,
        num_beam_groups=args.beam_group,
        num_beam_hyps_to_keep=args.num_beam_hyps_to_keep
    )

    # instantiate logits processors
    logits_processor = LogitsProcessorList([
        HammingDiversityLogitsProcessor(
            args.hamming_penalty, num_beams=args.beam_size, num_beam_groups=args.beam_group),
        MinLengthLogitsProcessor(
            args.min_len, eos_token_id=model.config.eos_token_id),
    ])

    outputs = model.group_beam_search(
        input_ids, beam_scorer, logits_processor=logits_processor, max_length=args.max_len, **model_kwargs)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_outputs


def run_bs(args, model, input_doc: str):
    encoder_input_ids = tokenizer(
        input_doc, return_tensors="pt").input_ids.to(args.device)

    # lets run diverse beam search using 6 beams
    num_beams = args.beam_size

    # define decoder start token ids
    input_ids = torch.ones(
        (num_beams, 1), device=model.device, dtype=torch.long)
    input_ids = input_ids * model.config.decoder_start_token_id
    model_kwargs = {
        "encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)}
    beam_scorer = BeamSearchScorer(
        batch_size=1,
        max_length=args.max_len,
        num_beams=args.beam_size,
        device=model.device,
        num_beam_hyps_to_keep=args.num_beam_hyps_to_keep
    )

    # instantiate logits processors
    logits_processor = LogitsProcessorList([
        MinLengthLogitsProcessor(
            args.min_len, eos_token_id=model.config.eos_token_id),
    ])

    outputs = model.beam_search(
        input_ids, beam_scorer, logits_processor=logits_processor, max_length=args.max_len, **model_kwargs)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_outputs


def run_topp(args, model, input_doc):
    input_ids = tokenizer(
        input_doc, return_tensors="pt").input_ids.to(args.device)
    outputs = []
    for i in range(args.num_beam_hyps_to_keep):
        output = model.generate(input_ids=input_ids, top_p=args.top_p,
                                do_sample=True, min_length=args.min_len, max_length=args.max_len)
        outputs.append(output[0].cpu().tolist())
    # print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_outputs


def run_recom(args, model, input_doc):
    input_ids = tokenizer(
        input_doc, return_tensors="pt").input_ids.to(args.device)
    output = recomb_beam_search(input_ids, model, pad_token_id=tokenizer.pad_token_id,
                       eos_token_id=tokenizer.eos_token_id, beam_sz=5, max_len=5, num_return_hypo=100)
    return output
def run_best(args, model, inp):
    input_ids = tokenizer(
        inp, return_tensors="pt").input_ids.to(args.device)
    output = best_first_search(input_ids, model, pad_token_id=tokenizer.pad_token_id,
                       eos_token_id=tokenizer.eos_token_id,  max_len=30)
    return output 
def main(args):
    eval_rouge, eval_bleu, eval_rep = [], [], []
    # logging.info(args)
    nexample = 20
    cnt = 0
    all_outputs = []
    all_summaries = []
    for example in dataset:
        cnt += 1
        document = example['document']
        sents = document.split('\n')
        inp = "\n".join(sents[:10])[:5000]

        doc_id = example['id']
        ref_sum = example['summary']
        logging.info(f"\n\n===Inp Doc: {document[:2000]}\n---Sum: {ref_sum}")
        if args.model == 'dbs':
            output = run_dbs(args, model, inp)
        elif args.model == 'bs':
            output = run_bs(args, model, inp)
        elif args.model == 'topp':
            output = run_topp(args, model, inp)
        elif args.model =='recom':
            output = run_recom(args, model, inp)
        elif args.model == 'best':
            output = run_best(args, model, inp)
        all_outputs.append(output)
        all_summaries.append(ref_sum)
        if cnt > nexample:
            break
    """
    for summ, out in zip(all_summaries, all_outputs):
        eval_rouge.append(rouge(out, summ))
        eval_rep.append(repetition(out))
        eval_bleu.append(self_bleu(out))
    eval_rouge = statistics.mean(eval_rouge)
    eval_rep = statistics.mean(eval_rep)
    eval_bleu = statistics.mean(eval_bleu)
    logging.info(f"{eval_rouge},{eval_rep},{eval_bleu}")
    """

if __name__ == "__main__":
    # execute only if run as a script
    args = process_arg()
    args.device = device
    setup_logger(name=f"{args.model}")
    main(args)
