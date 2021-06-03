path = '/mnt/data0/ojas/QFSumm/data/tokenized/cnndm_qf'
import os
import random
files = os.listdir(path)
files = [x for x in files if x.endswith('tokens')]
random.shuffle(files)
files = files[:100]
for f in files:
    with open(f"{path}/{f}",'r') as fd:
        lines = fd.read().splitlines()
    print(f"\t\t\t\t\t\t\t\t{f}")
    print(lines)
