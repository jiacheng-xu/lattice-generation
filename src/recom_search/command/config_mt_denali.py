from random import random
from subprocess import Popen

task = [
    " -task mt1n -dataset en-fr",
    " -task mt1n -dataset en-zh",
    " -task mtn1 -dataset fr-en",
    " -task mtn1 -dataset zh-en"
]
import random
cuda_range = [0,3,1,2]
i = 0

bag = []
cmd_base = "PYTHONPATH=./ python src/recom_search/command/run_pipeline.py -beam_size 10   "

ngram = " -ngram_suffix 4"
beam = " -beam_group 5"
length = " -min_len 3 -max_len -1"

cmd_base += ngram+beam+length


d = {}

avg_score =  [ -1, 0.5, 0.75]
model = [" -adhoc ", '-post -post_ratio 0.3 ', '-post -post_ratio 0.7 ']
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