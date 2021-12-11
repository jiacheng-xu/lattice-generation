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

# a star configs
# avg_score =  [-1, 0.75]
avg_score =  [0.75]
astar_mode = [" -adhoc "]
common_a_star = []
for score in avg_score:
    for m in astar_mode:
        common_a_star.append(f" -avg_score {score} {m} ")


base = "PYTHONPATH=./ python src/recom_search/command/run_pipeline.py -nexample 100   "

ngram = " -ngram_suffix 4 "
length = " -min_len 3 -max_len -1 "
beam = " -beam_group 4 -beam_size 8 "
low_beam = " -beam_group 1 -beam_size 2 "
mid_beam = " -beam_group 2 -beam_size 4 "

cmd_base =base+ ngram+beam+length
cmd_base_low = base+ ngram+low_beam+length
cmd_base_mid = base+ ngram+mid_beam+length

config_dbs =  ["-model dbs -hamming_penalty 2.0"]
config_bs = ["-model bs "]
config_topp = ["-model topp -top_p 0.8", "-model topp -top_p 0.9"]
config_greed = [" -model greedy"]
config_temp = [" -model temp -temp 1.5", " -model temp -temp 1.25"]
config_recom = [" -model recom_bs"]
config_recom_sample = [' -model recom_sample -top_p 0.8 ',' -model recom_sample -top_p 0.9 ']
config_astar_base = ['-model astar_base -avg_score -1  -adhoc ','-model astar_base -avg_score -1  -post -post_ratio 0.3 ']
baselines = config_dbs + config_bs + config_topp + config_greed + config_temp + config_recom +  config_recom_sample + config_astar_base
baselines = []      # ONLY a star


import time
# (I) Base baseline
all_commands = []
for b in baselines:
    for t in task:
        all_commands.append(cmd_base + f" {t} -device cuda:{cuda_range[i % 4]}  " + b)
        i += 1

# (II) recomb + best
for b in common_a_star:
    for t in task:
        """
        all_commands.append(cmd_base + f" {t} -model astar -merge imp -device cuda:{cuda_range[i % 4]}  " + b)
        i += 1
        all_commands.append(cmd_base + f" {t} -model astar -merge zip -device cuda:{cuda_range[i % 4]}  " + b)
        i += 1
        """
        all_commands.append(cmd_base_low + f" {t} -model astar -merge zip -device cuda:{cuda_range[i % 4]}  " + b)
        i += 1
        all_commands.append(cmd_base_mid + f" {t} -model astar -merge zip -device cuda:{cuda_range[i % 4]}  " + b)
        i += 1
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