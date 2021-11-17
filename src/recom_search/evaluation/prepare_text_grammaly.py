
fdir = '/mnt/data1/jcxu/lattice-sum/output/text'
tgt_dir = '/mnt/data1/jcxu/lattice-sum/output'
import json
import os
import statistics
from collections import defaultdict
import glob

folders = glob.glob(f"{fdir}/**/" ) # folders under stat fdir

import random

for fold_dir in folders:
    name = fold_dir.split('/')[-2:-1][0]
    print(name)
    files  = os.listdir(fold_dir)
    files = [f for f in files if f.endswith('txt')]
    if len(files) < 100:
        print(fold_dir, len(files))
        continue
    bag = []
    for f in files:
        with open(os.path.join(fold_dir, f),'r') as fd:
            lines = fd.read().splitlines()
        bag += lines
    random.shuffle(bag)
    bag = bag[:500]

    with open(os.path.join(tgt_dir, name+'.txt'),'w') as fd:
        fd.write("\n".join(bag))