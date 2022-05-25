
PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -model astar -dfs_expand -dataset fr-en -beam_size 2 -task mtn1 -min_len 5 -max_len -1 -ngram_suffix 4 -merge zip -device cuda:2 

PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -model astar -dfs_expand -dataset fr-en -beam_size 2 -task mtn1  -min_len 5 -max_len -1 -ngram_suffix 4 -merge zip -device cuda:3 -avg_score 0.5

PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -model astar -dfs_expand -dataset fr-en -beam_size 2 -task mtn1  -min_len 5 -max_len -1 -ngram_suffix 4 -merge zip -device cuda:0 -avg_score 1

PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -model astar -dataset fr-en -beam_size 2 -task mtn1  -max_len -1 -post -post_ratio 0.7  -device cuda:0 -ngram_suffix 4 -merge zip -min_len 5

PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -model astar -dataset fr-en -beam_size 2 -task mtn1  -max_len -1 -post -post_ratio 0.5  -device cuda:1 -ngram_suffix 4 -merge zip 

PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -model astar -dataset fr-en -beam_size 2 -task mtn1  -max_len -1 -post -post_ratio 0.3  -device cuda:1 -ngram_suffix 4 -merge zip -min_len 5


PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -model astar -dataset fr-en -beam_size 2 -task mtn1  -max_len -1 -post -post_ratio 0.7  -device cuda:2 -ngram_suffix 4 -merge zip -avg_score 0.5 -min_len 5

PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -model astar -dataset fr-en -beam_size 2 -task mtn1  -max_len -1 -post -post_ratio 0.5  -device cuda:2 -ngram_suffix 4 -merge zip -avg_score 0.5 -min_len 5

PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -model astar -dataset fr-en -beam_size 2 -task mtn1  -max_len -1 -post -post_ratio 0.3  -device cuda:2 -ngram_suffix 4 -merge zip -avg_score 0.5 -min_len 5


PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -model astar -dataset fr-en -beam_size 2 -task mtn1  -max_len -1 -post -post_ratio 0.7  -device cuda:0 -ngram_suffix 4 -merge zip -avg_score 1 -min_len 5

PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -model astar -dataset fr-en -beam_size 2 -task mtn1  -max_len -1 -post -post_ratio 0.5  -device cuda:3 -ngram_suffix 4 -merge zip -avg_score 1 -min_len 5

PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -model astar -dataset fr-en -beam_size 2 -task mtn1  -max_len -1 -post -post_ratio 0.3  -device cuda:3 -ngram_suffix 4 -merge zip -avg_score 1 -min_len 5