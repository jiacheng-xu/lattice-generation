fdir = '/mnt/data1/jcxu/back_to_fact/result'

import json

import os
import statistics
from collections import defaultdict
files  = os.listdir(fdir)
for f in files:
    summary = defaultdict(list)
    with open(os.path.join(fdir, f),'r') as fd:
        data = json.load(fd)
    for d in data:
        for k,v in d.items():
            summary[k].append(v)
    
    outputs = [f]
    print(";".join(list(summary.keys() )))
    for k, v in summary.items():
        if k == 'file':
            continue
        m = statistics.median(v)
        outputs.append(str(m))
    
    print(";".join(outputs))