# read
from subprocess import Popen
import random
import os
import subprocess
print(os.curdir)
with open('/mnt/data1/jcxu/back_to_fact/src/recom_search/command/run.sh', 'r') as fd:
    lines = fd.read().splitlines()
lines = [l for l in lines if len(l) > 10]
print(lines)
random.shuffle(lines)


# run commands in parallel
processes = [Popen(lines[i], shell=True)
             for i in range(len(lines))]
# collect statuses
exitcodes = [p.wait() for p in processes]
