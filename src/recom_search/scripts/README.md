`run.sh` collects some commands of models.

For baseline recomb model, run `PYTHONPATH=./ python src/recom_search/scripts/run_eval.py -model recom -beam_size 10 -ngram_suffix 3`. You can adjust the beam size and the ngram suffix matching. For other models, check the `process_arg` function in `run_eval.py`.
Logs are stored under `logs`; data generated in '*.pkl' are stored in `vizs`.

Visualization
Run `src/recom_search/evaluation/util.py` to render those HTML files. 



| Name | Description |
|-------|---------|
| dfs_expand | Use depth-first expansion or not. When running best-first search based algorithms, expand one hypothesis according to PQ and **always** greedily rollout the hypothesis until reaching EOS.  |
| max_len | Max decoding length. -1 means using 2* current input length as the max decoding length. This was mostly used in machine translation experiment. |
| k_best | Max number of next step prediction considered. LM will yield a prob distb of vocab_size, and we only put top k_best in the search frontier for future search and expansion. default=5. Set to save space. |
|avg_score | Co-efficient for model score computation. When set to -1, it means summing the log likelihood across time steps $\sum \log p_i$. Otherwise, avg_score $\alpha$ is used to calibrate sequence length and the model score is computed as $\frac{ \sum \log p_i }{ len(x)^{\alpha} } $. Typical choice of $\alpha$ is 0.6 or 0.8 or 1 (average).|
| | |