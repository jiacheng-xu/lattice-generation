
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model dbs -hamming_penalty 1.0 #dbs
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model bs -device cuda:0
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model topp -top_p 0.9 -device cuda:0 #topp

PYTHONPATH=./ python src/recom_search/command/run_eval.py -model recom_bs -device cuda:0

PYTHONPATH=./ python src/recom_search/command/run_eval.py -model recom_sample -device cuda:1

PYTHONPATH=./ python src/recom_search/command/run_eval.py -model astar -device cuda:1

PYTHONPATH=./ python src/recom_search/command/run_eval.py -model astar -use_heu -heu_seq_score_len_rwd 1. -device cuda:1  # automatic end

PYTHONPATH=./ python src/recom_search/command/run_eval.py -model astar -use_heu -heu_seq_score_len_rwd 0.5  -device cuda:3 # automatic end

PYTHONPATH=./ python src/recom_search/command/run_eval.py -model astar -post -post_ratio 0.7  -device cuda:3 # post end
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model astar -post -post_ratio 0.5  -device cuda:3 # post end

PYTHONPATH=./ python src/recom_search/command/run_eval.py -model astar -adhoc -device cuda:2  # intermediate end
