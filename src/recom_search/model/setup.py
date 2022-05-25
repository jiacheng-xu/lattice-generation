
import sys
import logging
from datasets import load_dataset
import argparse
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoConfig, AutoModelForSeq2SeqLM,AutoTokenizer
import os
import random

random.seed(2021)

def render_address(root = 'output') ->dict:
    """
    create name of subdirectories
    """
    d = {
        'data':os.path.join(root, 'data'),
        'html':os.path.join(root, 'html'),
        'stat':os.path.join(root, 'stat'),
        'text':os.path.join(root, 'text'),
        'table':os.path.join(root, 'table'),
    }
    return d

def read_mt_data(path='/mnt/data1/jcxu/lattice-sum/mt-data/use', name='zh-en'):
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


# MODEL_CACHE = '/mnt/data1/jcxu/cache'


def setup_model(task='sum', dataset='xsum', model_name='facebook/bart-large-xsum', device_name='cuda:2'):
    device = torch.device(device_name)
    print(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if task == 'custom':
        # you need to store the input under the path_dataset folder
        dec_prefix = [tokenizer.eos_token_id]
        with open(os.path.join(dataset, 'input.txt'), 'r') as fd:
            slines = fd.read().splitlines()
        with open(os.path.join(dataset, 'output.txt'), 'r') as fd:
            tlines = fd.read().splitlines()
        dataset = zip(slines, tlines)
    elif task == 'sum':
        logging.info('Loading dataset')
        if dataset == 'xsum':
            dataset = load_dataset("xsum", split='validation')
        elif dataset == 'cnndm':
            raise NotImplementedError("not supported")
            dataset = load_dataset("cnn_dailymail", split='validation')
            print("CNNDM mean token in ref 56")
        dec_prefix = [tokenizer.eos_token_id]
    elif task == 'mt1n':
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        model = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50-one-to-many-mmt")
        tokenizer = MBart50TokenizerFast.from_pretrained(
            "facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")
        assert dataset.startswith('en')
        tgt_lang = dataset[3:]
        dataset = read_mt_data(name=dataset)

        from transformers.models.mbart.tokenization_mbart import FAIRSEQ_LANGUAGE_CODES
        match = [x for x in FAIRSEQ_LANGUAGE_CODES if x.startswith(tgt_lang)]
        assert len(match) == 1
        lang = match[0]
        logging.info(f"Lang: {lang}")
        dec_prefix = [tokenizer.eos_token_id, tokenizer.lang_code_to_id[lang]]
        logging.info(f"{tokenizer.decode(dec_prefix)}")
    elif task == 'mtn1':
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        model = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50-many-to-one-mmt", )
        tokenizer = MBart50TokenizerFast.from_pretrained(
            "facebook/mbart-large-50-many-to-one-mmt")
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
        logging.info(f"{tokenizer.decode(dec_prefix)}")
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
    parser.add_argument('-device', type=str, default='cuda:2', help='name of device, eg. cuda:0 or cpu')
    parser.add_argument("-model", type=str, choices=[
                        'dbs', 'bs', 'greedy', 'topp', 'temp', 'bs_recom', 'sample_recom', 'bfs','bfs_recom'], default='bs')
    parser.add_argument('-beam_size', type=int, default=15)
    parser.add_argument('-nexample', type=int, default=100)

    parser.add_argument('-task', type=str, default='sum',
                        choices=['sum', 'mt1n', 'mtn1', 'custom'], help='for custom, you need to define your data IO')
    parser.add_argument('-dataset', default='xsum', type=str)
    parser.add_argument('-hf_model_name', default='facebook/bart-large-xsum', type=str)

    parser.add_argument('-path_output', type=str, default='custom_output')

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

    parser.add_argument('-avg_score', type=float, default=-1,
                        help='average model score coefficient. typical numbers like 0.6 or 0.8 or 0.9')

    parser.add_argument('-use_heu', type=str2bool, nargs='?',
                        const=True, default=False, help='our model: do we use heuristic')
    parser.add_argument('-post', type=str2bool, nargs='?',
                        const=True, default=False, help='our model: enforce the model to generate after exploration')
    parser.add_argument('-dfs_expand', type=str2bool, nargs='?',
                        const=True, default=False, help='our model: always generate till the end once touch a node')
    parser.add_argument('-post_ratio', type=float, default=0.4,
                        help='our model: ratio of resource allocation')
    
    # start of depricated
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
    # end of depricated
    parser.add_argument('-merge', type=str, default='zip',
                        choices=['zip', 'rcb'])

    args = parser.parse_args()
    return args

