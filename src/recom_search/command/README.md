`run.sh` collects some commands of models.

For baseline recomb model, run `PYTHONPATH=./ python src/recom_search/command/run_eval.py -model recom -beam_size 10 -ngram_suffix 3`. You can adjust the beam size and the ngram suffix matching. For other models, check the `process_arg` function in `run_eval.py`.
Logs are stored under `logs`; data generated in '*.pkl' are stored in `vizs`.

Visualization
Run `src/recom_search/evaluation/util.py` to render those HTML files. 