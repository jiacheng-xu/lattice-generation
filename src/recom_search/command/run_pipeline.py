"""
1. run model to get raw .pkl files (data)
2. analyze graph and yield sample paths and oracle paths (text)
3. visuliaze into html (html)
3. down stream evaluation on sample paths and oracle paths (stat)
"""


from src.recom_search.evaluation.deep_ana import deep_analyze_main
from src.recom_search.evaluation.analysis import analyze_data, analyze_main
from src.recom_search.model.util import render_config_name
from src.recom_search.command.run_eval import run_model

from multiprocessing import Pool
from src.recom_search.model.setup import tokenizer, model, dataset, dec_prefix, args, dict_io
import logging

if __name__ == '__main__':
    logging.info(f"Start running the pipeline")
    param_sim_function = {
        'ngram_suffix': args.ngram_suffix,
        'len_diff': args.len_diff,
        'merge': args.merge
    }
    config_search = {
        'post': args.post,
        'post_ratio': args.post_ratio,  # ratio of model calls left for post finishing
        'adhoc': args.adhoc,
        'heu': args.use_heu
    }
    combined_dict = {**config_search, **param_sim_function}
    combined_dict['avgsco'] = args.avg_score
    combined_dict['lenrwd'] = args.heu_seq_score_len_rwd
    combined_dict['topp'] = args.top_p
    config_name = render_config_name(
        args.task, args.dataset, args.model, args.beam_size, args.max_len, combined_dict)
    logging.info(f"Config name: {config_name}")
    run_model(args, tokenizer, model, dataset, dec_prefix, dict_io['data'])
    del model
    logging.info(f"Done with making data. Start analyzing data.")
    analyze_main(config_name, dict_io['data'], dict_io['text'], dict_io['stat'], dict_io['html'])
    logging.info("Done with initial analysis")
    # second stage analysis: run gector model, get number of finished nodes from data, analyze model parameter, gather results to json and a latex table
    deep_analyze_main(args, config_name, dict_io['data'], dict_io['text'], dict_io['stat'], dict_io['table'])
    logging.info("Done with all. bye bye")