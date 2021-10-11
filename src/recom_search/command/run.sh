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


PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best  -heu_seq_score 0.1  -heu_seq_score_len_rwd 0  -heu_pos 0  -heu_ent 0  -heu_word 0 
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best  -heu_seq_score 0.05  -heu_seq_score_len_rwd 0  -heu_pos 0  -heu_ent 0  -heu_word 0 
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best  -heu_seq_score 0.5  -heu_seq_score_len_rwd 0  -heu_pos 0  -heu_ent 0  -heu_word 0 
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best  -heu_seq_score 0  -heu_seq_score_len_rwd 0.01  -heu_pos 0  -heu_ent 0  -heu_word 0 

PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best  -heu_seq_score 0  -heu_seq_score_len_rwd 0.05  -heu_pos 0  -heu_ent 0  -heu_word 0 
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best  -heu_seq_score 0  -heu_seq_score_len_rwd 0.1  -heu_pos 0  -heu_ent 0  -heu_word 0 
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best  -heu_seq_score 0  -heu_seq_score_len_rwd 0  -heu_pos 0.1  -heu_ent 0  -heu_word 0 
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best  -heu_seq_score 0  -heu_seq_score_len_rwd 0  -heu_pos 1  -heu_ent 0  -heu_word 0 

PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best  -heu_seq_score 0  -heu_seq_score_len_rwd 0  -heu_pos 10  -heu_ent 0  -heu_word 0 
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best  -heu_seq_score 0  -heu_seq_score_len_rwd 0  -heu_pos 0  -heu_ent 0.5  -heu_word 0 
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best  -heu_seq_score 0  -heu_seq_score_len_rwd 0  -heu_pos 0  -heu_ent 1  -heu_word 0 
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best  -heu_seq_score 0  -heu_seq_score_len_rwd 0  -heu_pos 0  -heu_ent 5  -heu_word 0 

PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best  -heu_seq_score 0  -heu_seq_score_len_rwd 0  -heu_pos 0  -heu_ent 0  -heu_word 0.5 
PYTHONPATH=./ python src/recom_search/command/run_eval.py -model best  -heu_seq_score 0  -heu_seq_score_len_rwd 0  -heu_pos 0  -heu_ent 0  -heu_word 1 