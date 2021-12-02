from subprocess import Popen

base  = "PYTHONPATH=./ python src/recom_search/command/run_pipeline.py -nexample 100 "
cuda_range = [0,0,1,2]

# a star configs
avg_score =  [-1, 0.75]
avg_score =  [0.75]
astar_mode = [" -adhoc ",'-post -post_ratio 0.7 ','-post -post_ratio 0.3 ',]
common_a_star = []
for score in avg_score:
    for m in astar_mode:
        common_a_star.append(f" -avg_score {score} {m} ")

i = 0

bag = []

ngram = " -ngram_suffix 4"
beam = " -beam_group 4 -beam_size 16"
low_beam = " -beam_group 4 -beam_size 4"
length = " -min_len 10 -max_len 35"

cmd_base =base+ ngram+beam+length
cmd_base_low =base+ ngram+low_beam+length

config_dbs =  ["-model dbs -hamming_penalty 2.0"]
config_bs = ["-model bs "]
config_topp = ["-model topp -top_p 0.8", "-model topp -top_p 0.9"]
config_greed = [" -model greedy"]
config_temp = [" -model temp -temp 1.5", " -model temp -temp 1.25"]
config_recom = [" -model recom_bs"]
config_recom_sample = [' -model recom_sample -top_p 0.8 ',' -model recom_sample -top_p 0.9 ']
config_astar_base = ['-model astar_base -avg_score -1  -adhoc ','-model astar_base -avg_score -1  -post -post_ratio 0.3 ']
baselines = config_dbs + config_bs + config_topp + config_greed + config_temp + config_recom +  config_recom_sample + config_astar_base
baselines = []

# d = {'heu_seq_score': [0.0, 0.1, 0.05,0.5],
#      'heu_seq_score_len_rwd': [0.0,0.01, 0.05,0.1],
#      'heu_pos': [0.0,0.1, 1, 10], 'heu_ent': [0.0, 0.5, 1, 5],  'heu_word': [0.0,0.5, 1]
#      }
# default_d = {'heu_seq_score': 0,
#      'heu_seq_score_len_rwd': 0,
#      'heu_pos': 0, 'heu_ent': 0,  'heu_word': 0
#      }

import time
# (I) Base baseline
all_commands = []
for b in baselines:
    all_commands.append(cmd_base + f" -device cuda:{cuda_range[i % 4]}  " + b)
    i += 1

# (II) recomb + best
for b in common_a_star:
    all_commands.append(cmd_base + f" -model astar -merge imp -device cuda:{cuda_range[i % 4]}  " + b)
    i += 1
    all_commands.append(cmd_base + f" -model astar -merge zip -device cuda:{cuda_range[i % 4]}  " + b)
    i += 1
    all_commands.append(cmd_base_low + f" -model astar -merge zip -device cuda:{cuda_range[i % 4]}  " + b)
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
