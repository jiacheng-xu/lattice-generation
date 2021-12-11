from subprocess import Popen

base  = "PYTHONPATH=./ python src/recom_search/command/run_pipeline.py -nexample 100 "
cuda_range = [0,0,1,2]

i = 0

bag = []

ngram = " -ngram_suffix 4"
beams = []

length = " -min_len 10 -max_len 35"

for beam_size in [32,64,128]:
    beams.append(f" -beam_group {beam_size // 4} -beam_size {beam_size} ")
cmd_base =base+ ngram+length


config_dbs =  ["-model dbs -hamming_penalty 2.0"]
config_bs = ["-model bs "]
baselines = config_dbs + config_bs


import time
# (I) Base baseline
all_commands = []
for b in baselines:
    for beam in beams:
        all_commands.append(cmd_base + beam+ f" -device cuda:{cuda_range[i % 4]}  " + b)
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
