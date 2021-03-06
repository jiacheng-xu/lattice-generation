from random import random
from subprocess import Popen

task = [
    " -task mt1n -dataset en-fr",
    " -task mt1n -dataset en-zh",
    " -task mtn1 -dataset fr-en",
    " -task mtn1 -dataset zh-en"
]
import random
cuda_range = [0,0,1,2]
i = 0

bag = []
cmd_base = "PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -beam_size 10   "

ngram = " -ngram_suffix 4"
beam = " -beam_group 5"
length = " -min_len 3 -max_len -1"

cmd_base += ngram+beam+length


config_dbs =  ["-model dbs -hamming_penalty 2.0"]

config_bs = ["-model bs "]
config_topp = ["-model topp -top_p 0.8", "-model topp -top_p 0.9"]
config_greed = [" -model greedy"]
config_temp = [" -model temp -temp 1.5"]
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
    for t in task:
        final_bases.append(cmd_base + f" -device cuda:{cuda_range[i % 4]}  " + b + t)
        i += 1

# run commands in parallel
import time

random.shuffle(final_bases)
print("We are going to run many commands, they are:")
print("\n".join(final_bases))
print('waiting')


time.sleep(5)
processes = []
for i in range(len(final_bases)):
    p = Popen(final_bases[i], shell=True)
    processes.append(p)
    time.sleep(60)
exitcodes = [p.wait() for p in processes]
exit()
# (II) Recomb baseline

d = {}

avg_score =  [ -1, 0.5, 0.75]
model = [" -dfs_expand ", '-post -post_ratio 0.3 ', '-post -post_ratio 0.7 ']
models = []
for score in avg_score:
    for m in model:
        models.append(f" -avg_score {score} {m} ")
final_bases = []
for b in models:
    for t in task:
        final_bases.append(cmd_base + f" -model astar -device cuda:{cuda_range[i % 4]}  " +t + b)
    i += 1

# run commands in parallel
import time
print("We are going to run many commands, they are:")
print("\n".join(final_bases))
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