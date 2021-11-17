
fdir = '/mnt/data1/jcxu/lattice-sum/output/stat'
import json

import os
import statistics
from collections import defaultdict
import glob

folders = glob.glob(f"{fdir}/**/" ) # folders under stat fdir
# print(folders)
prints = []

all_result = []
all_keys = ["name"]
for fold_dir in folders:
    # fold_dir = os.path.join(fdir, fold)
    files  = os.listdir(fold_dir)
    if len(files) < 100:
        print(fold_dir, len(files))
        continue
    summary = defaultdict(list)
    for f in files:
        
        with open(os.path.join(fold_dir, f),'r') as fd:
            data = json.load(fd)
        
        for k,v in data.items():
            summary[k].append(v)
    
    outputs = defaultdict(str)
    
    for k, v in summary.items():
        
        if k == 'file':
            # outputs.append(v[0])
            continue
        if k not in all_keys:
            all_keys.append(k)
        m = statistics.mean(v)
        outputs[k] = str(m)
    outputs['name'] = fold_dir
    all_result.append(outputs)

pad_result = [[] for _ in range(len(all_result))]
all_keys =[ x for x in all_keys if ('REP' not in x and 'SELF' not in x)]
for k in all_keys:
    for idx, res in enumerate(all_result):
        if k in res:
            pad_result[idx].append(res[k])
        else:
            pad_result[idx].append(" ")
keys = ";".join(all_keys)
print(keys)
for p in pad_result:
    print(";".join(p))