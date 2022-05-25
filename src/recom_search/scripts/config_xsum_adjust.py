from subprocess import Popen

base  = "PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -nexample 100 "
cuda_range = [0,0,1,2]

# a star configs
avg_score =  [0.75]

i = 0

bag = []

ngram = " -ngram_suffix 4"
beam = " -beam_group 4 -beam_size 17"
length = " -min_len 10 -max_len 35"

cmd_base =base+ ngram+beam+length

config_dbs =  ["-model dbs -hamming_penalty 2.0"]
config_bs = ["-model bs "]
config_topp = ["-model topp -top_p 0.8", "-model topp -top_p 0.9"]
# config_greed = [" -model greedy"]
config_temp = [" -model temp -temp 1.5"]
config_recom = [" -model recom_bs"]
# config_recom_sample = [' -model recom_sample -top_p 0.8 ',' -model recom_sample -top_p 0.9 ']
# config_astar_base = ['-model astar_base -avg_score -1  -dfs_expand ','-model astar_base -avg_score -1  -post -post_ratio 0.3 ']
baselines = config_dbs + config_bs + config_topp +  config_temp + config_recom 


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
    time.sleep(30)
exitcodes = [p.wait() for p in processes]
