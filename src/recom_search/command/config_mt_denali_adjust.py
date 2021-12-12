from random import random
from subprocess import Popen

task = [
    " -task mt1n -dataset en-fr",
    " -task mt1n -dataset en-zh",
    " -task mtn1 -dataset fr-en",
    " -task mtn1 -dataset zh-en"
]


cuda_range = [0,3,1,2]
i = 0

bag = []


base = "PYTHONPATH=./ python src/recom_search/command/run_pipeline.py -nexample 100   "

ngram = " -ngram_suffix 4 "
length = " -min_len 3 -max_len -1 "
# beam = " -beam_group 4 -beam_size 8 "
beam = " -beam_group 4 -beam_size 12 "

cmd_base =base+ ngram+beam+length

config_dbs =  ["-model dbs -hamming_penalty 2.0"]
config_bs = ["-model bs "]
config_topp = ["-model topp -top_p 0.8", "-model topp -top_p 0.9"]
config_temp = [" -model temp -temp 1.5"]
config_recom = [" -model recom_bs"]
# config_recom_sample = [' -model recom_sample -top_p 0.8 ',' -model recom_sample -top_p 0.9 ']
# config_astar_base = ['-model astar_base -avg_score -1  -adhoc ','-model astar_base -avg_score -1  -post -post_ratio 0.3 ']
baselines = config_dbs + config_bs + config_topp  + config_temp + config_recom
# baselines = []      # ONLY a star


import time
# (I) Base baseline
all_commands = []
for b in baselines:
    for t in task:
        all_commands.append(cmd_base + f" {t} -device cuda:{cuda_range[i % 4]}  " + b)
        i += 1

# # (II) recomb + best
# for b in common_a_star:
#     for t in task:
#         """
#         all_commands.append(cmd_base + f" {t} -model astar -merge imp -device cuda:{cuda_range[i % 4]}  " + b)
#         i += 1
#         all_commands.append(cmd_base + f" {t} -model astar -merge zip -device cuda:{cuda_range[i % 4]}  " + b)
#         i += 1
#         """
#         all_commands.append(cmd_base_low + f" {t} -model astar -merge zip -device cuda:{cuda_range[i % 4]}  " + b)
#         i += 1
#         all_commands.append(cmd_base_mid + f" {t} -model astar -merge zip -device cuda:{cuda_range[i % 4]}  " + b)
#         i += 1
import random
random.shuffle(all_commands)
print("We are going to run many commands, they are:")
print("\n".join(all_commands))
print('waiting')
time.sleep(10)
processes = []
for i in range(len(all_commands)):
    p = Popen(all_commands[i], shell=True)
    processes.append(p)
    time.sleep(60)

exitcodes = [p.wait() for p in processes]