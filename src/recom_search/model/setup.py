import sys
import logging
from datasets import load_dataset
import argparse
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import os


def read_mt_data(path='/mnt/data1/jcxu/mt-data/wmt19', name='zh-en'):
    src = name[:2]
    tgt = name[3:]
    with open(os.path.join(path, f"{name}.{src}"), 'r') as fd:
        slines = fd.read().splitlines()
    with open(os.path.join(path, f"{name}.{tgt}"), 'r') as fd:
        tlines = fd.read().splitlines()
    print(slines[:5])
    print(tlines[:5])
    assert len(slines) == len(tlines)
    return zip(slines, tlines)


MODEL_CACHE = '/mnt/data1/jcxu/cache'


def setup_model(task='sum', dataset='xsum', device_name='cuda:2'):

    device = torch.device(device_name)
    if task == 'sum':
        model_name = 'facebook/bart-large-xsum'
        tokenizer = BartTokenizer.from_pretrained(
            model_name, cache_dir=MODEL_CACHE)

        logging.info('Loading model')
        model = BartForConditionalGeneration.from_pretrained(
            model_name, cache_dir=MODEL_CACHE)

        logging.info('Loading dataset')
        if dataset == 'xsum':
            dataset = load_dataset("xsum", split='validation')
        elif dataset == 'cnndm':
            dataset = load_dataset("cnn_dailymail", split='validation')
            print("CNNDM mean token in ref 56")
        dec_prefix = [tokenizer.eos_token_id]

    elif task == 'mt1n':
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        model = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50-one-to-many-mmt", cache_dir=MODEL_CACHE)
        tokenizer = MBart50TokenizerFast.from_pretrained(
            "facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX", cache_dir=MODEL_CACHE)
        assert dataset.startswith('en')
        tgt_lang = dataset[3:]
        dataset = read_mt_data(name=dataset)

        from transformers.models.mbart.tokenization_mbart import FAIRSEQ_LANGUAGE_CODES
        match = [x for x in FAIRSEQ_LANGUAGE_CODES if x.startswith(tgt_lang)]
        assert len(match) == 1
        lang = match[0]
        dec_prefix = [tokenizer.eos_token_id, tokenizer.lang_code_to_id[lang]]

    elif task == 'mtn1':
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        model = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50-many-to-one-mmt", cache_dir=MODEL_CACHE)
        tokenizer = MBart50TokenizerFast.from_pretrained(
            "facebook/mbart-large-50-many-to-one-mmt", cache_dir=MODEL_CACHE)
        # dataset should be like "xx-en"
        assert dataset.endswith('-en')
        src_lang = dataset[:2]
        from transformers.models.mbart.tokenization_mbart import FAIRSEQ_LANGUAGE_CODES
        match = [x for x in FAIRSEQ_LANGUAGE_CODES if x.startswith(src_lang)]
        assert len(match) == 1
        lang = match[0]
        tokenizer.src_lang = lang
        dataset = read_mt_data(name=dataset)
        dec_prefix = [tokenizer.eos_token_id,
                      tokenizer.lang_code_to_id["en_XX"]]

    model = model.to(device)
    return tokenizer, model, dataset, dec_prefix


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_logger(name):
    import datetime
    now_time = datetime.datetime.now()
    logname = f"logs/{name}{str(now_time)[:16]}.txt"
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('- %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    file_handler = logging.FileHandler(logname, 'w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


def process_arg():

    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=str, default='cuda:2')
    parser.add_argument("-model", type=str, choices=[
                        'dbs', 'bs', 'greedy', 'topp', 'temp', 'recom_bs', 'recom_sample',  'astar'], default='bs')
    parser.add_argument('-beam_size', type=int, default=15)
    parser.add_argument('-nexample', type=int, default=100)
    parser.add_argument('-task', type=str, default='sum',
                        choices=['sum', 'mt1n', 'mtn1'])
    parser.add_argument('-dataset', default='xsum', type=str)
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

    parser.add_argument('-avg_score', type=float, default=-1, help='average model score coefficient. typical numbers like 0.6 or 0.8 or 0.9')

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
    parser.add_argument('-merge', type=str, default='zip',
                        choices=['zip', 'imp'])
    # parser.add_argument('-min_path', type=int, default=0,help='Bool indicator of if min_path or not')

    # parser.add_argument("-beam_ent", type=str2bool, nargs='?', const=True,default=False, help="Use entropy to dynamically operate beam.")
    args = parser.parse_args()
    return args


args = process_arg()
setup_logger(name=f"{args.model}")
tokenizer, model, dataset, dec_prefix = setup_model(
    args.task, args.dataset, args.device)
