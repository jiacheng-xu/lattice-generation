"PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best -heu_seq_score 0 -heu_seq_score_len_rwd 0 -heu_pos 0 -heu_ent 0.3 -heu_word 0.0"

from random import random

from subprocess import Popen

cmd_base = "PYTHONPATH=./ python src/recom_search/command/run_pipeline.py  "
cuda_range = [0,0,1,2]
i = 0
import random
bag = []

ngram = " -ngram_suffix 4"
beam = " -beam_group 5 -beam_size 20"
length = " -min_len 10 -max_len 35"

cmd_base += ngram+beam+length
config_dbs =  ["-model dbs -hamming_penalty 2.0", "-model dbs -hamming_penalty 1.0", "-model dbs -hamming_penalty 0.5"]

config_bs = ["-model bs "]
config_topp = ["-model topp -top_p 0.8", "-model topp -top_p 0.9"]
config_greed = [" -model greedy"]
config_temp = [" -model temp -temp 1.5", " -model temp -temp 1.25"]
config_recom = [" -model recom_bs"]
config_recom_sample = ['-model recom_sample']

baselines = config_dbs + config_bs + config_topp + config_greed + config_temp + config_recom +  config_recom_sample

# d = {'heu_seq_score': [0.0, 0.1, 0.05,0.5],
#      'heu_seq_score_len_rwd': [0.0,0.01, 0.05,0.1],
#      'heu_pos': [0.0,0.1, 1, 10], 'heu_ent': [0.0, 0.5, 1, 5],  'heu_word': [0.0,0.5, 1]
#      }
# default_d = {'heu_seq_score': 0,
#      'heu_seq_score_len_rwd': 0,
#      'heu_pos': 0, 'heu_ent': 0,  'heu_word': 0
#      }


# (I) Base baseline
final_bases = []
for b in baselines:
    final_bases.append(cmd_base + f" -device cuda:{i % 3}  " + b)
    i += 1

# run commands in parallel
import time


print("We are going to run many commands, they are:")
print(final_bases)
print('waiting')
time.sleep(10)
processes = []
for i in range(len(final_bases)):
    p = Popen(final_bases[i], shell=True)
    processes.append(p)
    time.sleep(60)
# collect statuses
exitcodes = [p.wait() for p in processes]

# (II) Recomb baseline

d = {}
d['-avg_score']= [-1,0.5, 0.75, 1.0]
avg_score =  [-1,0.5, 0.75, 1.0]
model = [" -adhoc ", '-post -post_ratio 0.3 ', '-post -post_ratio 0.5 ', '-post -post_ratio 0.7 ']
models = []
for score in avg_score:
    for m in model:
        models.append(f" -avg_score {score} {m} ")
final_bases = []
for b in models:
    final_bases.append(cmd_base + f" -model astar -device cuda:{cuda_range[i % 4]}  " + b)
    i += 1

# run commands in parallel
import time
print("We are going to run many commands, they are:")
print(final_bases)
print('waiting')
time.sleep(10)
processes = []
for i in range(len(final_bases)):
    p = Popen(final_bases[i], shell=True)
    processes.append(p)
    time.sleep(60)
    # if i % 4 == 0:
    #     time.sleep(360)
# collect statuses
exitcodes = [p.wait() for p in processes]