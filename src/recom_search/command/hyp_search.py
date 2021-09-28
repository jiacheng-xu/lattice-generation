"PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best -heu_seq_score 0 -heu_seq_score_len_rwd 0 -heu_pos 0 -heu_ent 0.3 -heu_word 0.0"
d = {'heu_seq_score': [0.0, 0.1, 0.05,0.5],
     'heu_seq_score_len_rwd': [0.0,0.01, 0.05,0.1],
     'heu_pos': [0.0,0.1, 1, 10], 'heu_ent': [0.0, 0.5, 1, 5],  'heu_word': [0.0,0.5, 1]
     }
default_d = {'heu_seq_score': 0,
     'heu_seq_score_len_rwd': 0,
     'heu_pos': 0, 'heu_ent': 0,  'heu_word': 0
     }
prefix = "PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best "
for k,v in d.items():
    tar = d[k]
    for x in v:
        if x == 0.0:
            continue
        copy_default_d = default_d.copy()
        copy_default_d[k] = x
        
        keys, values = list(copy_default_d.keys()), list(copy_default_d.values())
        tmp = [f" -{m} {n} " for m,n in zip(keys, values)]
        command = prefix + "".join(tmp)
        print(command)

