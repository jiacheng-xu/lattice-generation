
import argparse
import logging
import os
import pickle
import random
import statistics
import sys
from datetime import datetime
from typing import Dict, List
import multiprocessing
import torch
from datasets import load_dataset
from transformers import BartForConditionalGeneration, BartModel, BartTokenizer
import numpy as np
import pandas as pd

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

kld = torch.nn.KLDivLoss(log_target=True, reduction='none')

now = datetime.now()

logger = logging.getLogger('sum')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(f"{now.strftime('%m')}{now.strftime('%d')}.html")
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('<br>%(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)


def load_pickle(dir, fname) -> Dict:
    with open(os.path.join(dir, fname), 'rb') as rfd:
        data = pickle.load(rfd)
    return data


def pnum(num):
    return "{:.2f}".format(num)


def add_dataname_to_suffix(args, args_dir) -> str:
    out = f"{args_dir}_{args.data_name}"
    out = add_temp_to_suffix(args, out)
    if not os.path.exists(out):
        os.makedirs(out)
    return out


def add_temp_to_suffix(args, args_dir) -> str:
    out = f"{args_dir}_{args.temp}"
    # if not os.path.exists(out):
    # os.makedirs(out)
    return out


def dec_print_wrap(func):
    def wrapper(*args, **kwargs):
        logging.info("=" * 20)
        out = func(*args, **kwargs)
        logging.info("-" * 20)
        return out
    return wrapper


def read_meta_data(dir, fname):
    file_package = load_pickle(dir, fname)
    data: List = file_package['data']
    meta = file_package['meta']
    return data, meta


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

flatten = lambda t: [item for sublist in t for item in sublist]

def common_args():

    task_choice = ['inp_grad', 'int_grad', 'random', 'occ', 'lead','attn','sent']
    eval_mode = ['sel_tok', 'rm_tok', 'sel_sent', 'rm_sent']
    settings = ['all','ctx', 'novel', 'fusion', 'lm', 'hard']

    parser = argparse.ArgumentParser()
    parser.add_argument("-model_family", default='bart')
    parser.add_argument("-data_name", default='xsum', help='name of dataset')
    parser.add_argument("-mname_lm", default='facebook/bart-large')
    parser.add_argument("-mname_sum", default='facebook/bart-large-xsum')
    parser.add_argument('-truncate_sent', default=15,
                        help='the max sent used for perturbation')
    parser.add_argument('-truncate_word', default=70,
                        help='the max token in each single sentence')
    parser.add_argument(
        '-dir_meta', default="/mnt/data0/jcxu/meta_pred", help="The location to meta data.")
    parser.add_argument('-dir_base', default="/mnt/data0/jcxu/output_base")
    parser.add_argument('-dir_stat', default="/mnt/data0/jcxu/csv")

    parser.add_argument("-debug", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate debug mode.")

    parser.add_argument("-sent_pre_sel", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Use sentence to filter the input document.")

    parser.add_argument("-task", dest='task', choices=task_choice)
    parser.add_argument("-device", help="device to use", default='cuda:0')
    parser.add_argument('-max_example', default=5000,
                        help='The max number of examples (documents) to look at.')
    parser.add_argument('-num_run_cut', default=40)
    parser.add_argument('-batch_size', default=100, type=int)
    parser.add_argument('-eval_mode', dest='eval_mode', choices=eval_mode)
    parser.add_argument('-temp', type=float, default=0.5)
    parser.add_argument('-eval_set',dest='eval_set',choices=settings)
    parser.add_argument('-hard_max_len',default=500,type=int)
    return parser
    
import time
from operator import itemgetter





def show_top_k(prob, tokenizer,name="", prefix="", k=5):
    prob = prob.squeeze()
    topk_v, top_idx = torch.topk(prob, k=k)
    index = top_idx.tolist()
    toks = [tokenizer.convert_ids_to_tokens(i) for i in index]

    logger.info(f"Type: {name}")
    result = []
    for i, t in enumerate(toks):
        logger.info(f"{i}: {pnum(topk_v[i].item())} {prefix}{t}")
        result.append((pnum(topk_v[i].item()), t))
    return result


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def prepare_filtered_input_document(output_base_step, sent_token_ids, num_sent=2):

    single_distb = output_base_step['pert_distb']
    if 'pert_comb' in output_base_step:
        if output_base_step['pert_top1_double'] < output_base_step['pert_top']:
           
            sorted_index = argsort(single_distb)[::-1]
            use_index = sorted_index[:num_sent] 
        else:
            pert_comb = output_base_step['pert_comb']
            pert_double_distb = output_base_step['pert_distb_double']
            sorted_combo = argsort(pert_double_distb)[::-1][0]
            use_index = pert_comb[sorted_combo]

    else:

        sorted_index = argsort(single_distb)[::-1]
        use_index = sorted_index[:num_sent]
    use_sent = [sent_token_ids[idx] for idx in use_index]
    input_doc = [0]+ flatten(use_sent) +[2]

    # return new document (list start 0 end 2), and the index of the selected sentences
    return input_doc


def fix_args(args):
    args.dir_base = add_dataname_to_suffix(args, args.dir_base)
    args.dir_meta = add_dataname_to_suffix(args, args.dir_meta)
    args.dir_stat = add_dataname_to_suffix(args, args.dir_stat)
    if args.debug:
        args.device = 'cpu'
        args.mname_sum = 'sshleifer/distilbart-xsum-6-6'
    if hasattr(args, 'task'):
        if args.sent_pre_sel == True:
            args.task = f"{args.task}_sent_sel"

        args.dir_task = f"/mnt/data0/jcxu/task_{args.task}"
        args.dir_task = add_dataname_to_suffix(args, args.dir_task)
        args.dir_eval_save = f"/mnt/data0/jcxu/eval_{args.task}_{args.eval_mode}"
        args.dir_eval_save = add_dataname_to_suffix(args, args.dir_eval_save)
    return args


random.seed(2021)