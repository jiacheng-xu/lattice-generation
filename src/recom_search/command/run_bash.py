# read
from subprocess import Popen
import random
import os
import subprocess


# config for MT: -dataset fr-en -beam_size 2 -task mtn1 -min_len 7 -max_len 50

print(os.curdir)
with open('/mnt/data1/jcxu/back_to_fact/src/recom_search/command/run.sh', 'r') as fd:
    lines = fd.read().splitlines()
lines = [l for l in lines if len(l) > 10 and (not l.startswith('#'))]
print(lines)
random.shuffle(lines)
import time


# run commands in parallel
processes = []
for i in range(len(lines)):
    p = Popen(lines[i], shell=True)
    processes.append(p)
    time.sleep(60)
# collect statuses
exitcodes = [p.wait() for p in processes]
