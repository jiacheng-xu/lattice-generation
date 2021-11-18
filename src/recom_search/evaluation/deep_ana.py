import random
import pickle
from pathlib import Path
import os

import logging
import statistics
from subprocess import Popen

from src.recom_search.model.model_output import SearchModelOutput

logger = logging.getLogger(__name__)


def call_gector(config_name, texts, nsamples):
    pass


"""
$1 = /mnt/data1/jcxu/gector
$2 = input
$3 = output
$4 = cnt file
cd $1
conda activate run-gector
python predict.py --model_path  roberta_1_gectorv2.th \
                   --input_file $2 --output_file $3 --output_cnt_file $4
conda deactivate
"""


def deep_ana(f, config_name, dict_io_data,
             dict_io_text, dict_io_table, dict_io_stat):
    name = ".".join(f.split('.')[:-1])
    fname = os.path.join(dict_io_table, config_name, f"{name}.json")
    if os.path.isfile(fname):
        logging.info(f"File exist")
        return
    # summarization or translation?
    if config_name.startswith('sum'):
        flag_sum = True
    elif config_name.startswith('mt'):
        flag_sum = False
    else:
        raise ValueError("Task either sum or mt")

    with open(os.path.join(dict_io_data, config_name, f), 'rb') as fd:
        finished: SearchModelOutput = pickle.load(fd)
    logger.info(f)
    logger.info(finished.ends)
    if not finished.ends:
        return
    with open(os.path.join(dict_io_text, config_name, f"{name}.txt"), 'rb') as fd:
        fd.read().splitlines()

import glob
import json
from collections import defaultdict
def get_jsons_from_a_folder(folder_dir):
    files  = os.listdir(folder_dir)
    if len(files) < 100:
        print(folder_dir, len(files))
    
    content = defaultdict(list)
    for f in files:
        with open(os.path.join(folder_dir,f),'r') as fd:
            t = json.load(fd)
        for k, v in t.items():
            content[k].append(v)
    return content

def get_finished_hypo(folder_dir,files):
    
    es = []
    for f in files:
        with open(os.path.join(folder_dir,f),'rb') as fd:
            finished: SearchModelOutput  = pickle.load(fd)
        es.append(len(finished.ends))
    
        
    return statistics.mean(es)

def evaluate_grammar_gector(all_files, config_name, dict_io_text, dict_io_table, nexamples=1000, model_repo='/mnt/data1/jcxu/gector', proj_dir='/mnt/data1/jcxu/lattice-sum/'):
    file_txt = [".".join(f.split('.')[:-1])+'.txt' for f in all_files]
    all_lines = []
    for f in file_txt:
        with open(os.path.join(dict_io_text, config_name, f)) as rfd:
            lines = rfd.read().splitlines()
            lines = [l.strip() for l in lines]
            all_lines += lines
    random.shuffle(all_lines)
    all_lines = all_lines[:nexamples]
    input_file = os.path.join(proj_dir,dict_io_table, config_name, "input.txt")
    outpuf_file = os.path.join(proj_dir,dict_io_table, config_name, "output.txt")
    outpuf_cnt_file = os.path.join(proj_dir,
        dict_io_table, config_name, "output_cnt.txt")
    with open(input_file, 'w') as fd:
        fd.write('\n'.join(all_lines))

    command = f"sh src/recom_search/evaluation/bash_gector.sh {model_repo} {input_file} {outpuf_file} {outpuf_cnt_file}"
    print(command)
    p = Popen(command, shell=True)
    
    p.wait()
    os.chdir(proj_dir)
    with open(outpuf_cnt_file, 'r') as fd:
        lines = fd.read().splitlines()
    assert len(lines) == 1
    err = float(lines[0])
    return err
import csv
def deep_analyze_main(args, config_name, dict_io_data, dict_io_text, dict_io_stat, dict_io_table):
    raw_files = os.listdir(os.path.join(dict_io_data, config_name))
    # get number of finished nodes from data, analyze model parameter, gather results to json and a latex table
    Path(os.path.join(dict_io_table, config_name)).mkdir(
        parents=True, exist_ok=True)
    l = len(raw_files)
    if args.dataset.startswith('en'):
        error_rate = 0
    else:
        error_rate = evaluate_grammar_gector(raw_files, config_name,
                            dict_io_text, dict_io_table)
    num_ends = get_finished_hypo(os.path.join(dict_io_data, config_name), raw_files)
    data_points:dict = get_jsons_from_a_folder(os.path.join(dict_io_stat, config_name) )
    
    final = {}

    # keys in args
    arg_keys =[ 'task', 'dataset', 'model', 'beam_size', 'max_len'] +  ['ngram_suffix','len_diff','merge','post_ratio','adhoc','avg_score','heu_seq_score_len_rwd','top_p']
    for key in arg_keys:
        final[key] = getattr(args,key)
    final['error'] = error_rate
    final['num_end'] = num_ends
    for k, v in data_points.items():
        if k == 'file' or 'REP' in k:
            continue
        val = statistics.mean(v) if v else 0
        final[k] = "{:.4f}".format(val)
    gather_csv = os.path.join(dict_io_table, f"{args.task}_{args.dataset}.csv")
    add_key = False
    if not os.path.isfile(gather_csv):
        add_key = True
    with open(gather_csv, 'a') as csvfile:
        keys = list(final.keys())
        values = list(final.values())
        # values = [ "{:.4f}".format(v) for v in values]
        spamwriter = csv.writer(csvfile, delimiter=';')
        if add_key:
            spamwriter.writerow(keys)
        spamwriter.writerow(values)
