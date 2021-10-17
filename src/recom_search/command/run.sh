PYTHONPATH=./ python src/recom_search/command/run_eval.py -model dbs -hamming_penalty 1.0 #dbs
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model bs
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model topp -top_p 0.7 #topp

PYTHONPATH=./ python src/recom_search/command/run_eval.py -model dbs -hamming_penalty 2.0 #dbs
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model dbs -hamming_penalty 1.0 #dbs
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model dbs -hamming_penalty 0.2 #dbs
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model dbs -hamming_penalty 0.5 #dbs

PYTHONPATH=./ python src/recom_search/command/run_eval.py -model bs #bs
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model topp -top_p 0.5 #topp
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model topp -top_p 0.7 #topp
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model topp -top_p 0.9 #topp

PYTHONPATH=./ python src/recom_search/command/run_eval.py -model greedy # greedy
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model temp -temp 1.5 # temp
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model temp -temp 1.2 # temp
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model temp -temp 1.1 # temp


PYTHONPATH=./ python src/recom_search/command/run_eval.py -model recom # recom
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best -extra_steps 5 # best 
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best -extra_steps 10 # best 


PYTHONPATH=./ python src/recom_search/command/run_eval.py -model recom -beam_size 5 -ngram_suffix 3 # recom
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model recom -beam_size 5 -ngram_suffix 4 # recom
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model recom -beam_size 5 -ngram_suffix 5 # recom


PYTHONPATH=./ python src/recom_search/command/run_eval.py -model recom -beam_size 10 -ngram_suffix 3 # recom
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model recom -beam_size 10 -ngram_suffix 4 # recom
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model recom -beam_size 10 -ngram_suffix 5 # recom


PYTHONPATH=./ python src/recom_search/command/run_eval.py -model recom -beam_size 20 -ngram_suffix 3 # recom
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model recom -beam_size 20 -ngram_suffix 4 # recom
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model recom -beam_size 20 -ngram_suffix 5 # recom


PYTHONPATH=./ python src/recom_search/command/run_eval.py -model recom -beam_size 50 -ngram_suffix 3 # recom
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model recom -beam_size 50 -ngram_suffix 4 # recom
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model recom -beam_size 50 -ngram_suffix 5 # recom


PYTHONPATH=./ python src/recom_search/command/run_eval.py -model recom_bs

PYTHONPATH=./ python src/recom_search/command/run_eval.py -model recom_sample




PYTHONPATH=./ python src/recom_search/command/run_eval.py -model astar -use_heu -heu_seq_score_len_rwd 1.   # automatic end

PYTHONPATH=./ python src/recom_search/command/run_eval.py -model astar -post -post_ratio 0.7   # post end

PYTHONPATH=./ python src/recom_search/command/run_eval.py -model astar -adhoc   # intermediate end

