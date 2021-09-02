PYTHONPATH=./ python src/recom_search/command/run_eval.py -model dbs -hamming_penalty 1.0 #dbs
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model bs #bs
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model topp -top_p 0.7 #topp
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model greedy # greedy
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model temp -temp 1.5 # temp


PYTHONPATH=./ python src/recom_search/command/run_eval.py -model recom # recom
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best # best