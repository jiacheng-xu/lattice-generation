
fdir = '/mnt/data1/jcxu/lattice-sum/output/stat'
import json

import os
import statistics
from collections import defaultdict
import glob

folders = glob.glob(f"{fdir}/**/" ) # folders under stat fdir
# print(folders)
for fold_dir in folders:
    # fold_dir = os.path.join(fdir, fold)
    files  = os.listdir(fold_dir)
    for f in files:
        summary = defaultdict(list)
        with open(os.path.join(fold_dir, f),'r') as fd:
            data = json.load(fd)
        
        for k,v in data.items():
            summary[k].append(v)
    
    outputs = []
    
    for k, v in summary.items():
        if k == 'file':
            # outputs.append(v[0])
            continue
        m = statistics.mean(v)
        outputs.append(str(m))
    outputs.append(fold_dir)
    print(";".join(outputs))
print(";".join(list(summary.keys() )))
