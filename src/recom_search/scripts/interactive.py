import sys
from src.recom_search.model.model_output import SearchModelOutput
from src.recom_search.model.model_bfs import bfs

from src.recom_search.model.setup import setup_model
from src.recom_search.model.setup import process_arg
def main() -> int:
    """
    Required input: 
    """

    input_text = "Transformers provides APIs to easily download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save you time from training a model from scratch. The models can be used across different modalities such as: Text: text classification, information extraction, question answering, summarization, translation, and text generation in over 100 languages. Images: image classification, object detection, and segmentation. Our library supports seamless integration between three of the most popular deep learning libraries: PyTorch, TensorFlow and JAX. Train your model in three lines of code in one framework, and load it for inference with another."
    args = process_arg()
    args.task = 'sum'
    args.dataset = 'xsum'

    args.task = 'custom'
    args.dataset = 'custom_input'

    # args.hf_model_name = 'facebook/bart-large-xsum'
    # args.hf_model_name = 'sshleifer/distilbart-cnn-6-6'
    tokenizer, model, dataset, dec_prefix = setup_model(args.task, args.dataset, args.hf_model_name,  args.device)

    param_sim_function = {
            'ngram_suffix': args.ngram_suffix,
            'len_diff': args.len_diff,
            'merge': args.merge
        }
    config_search = {
            'post': args.post,
            'post_ratio': args.post_ratio,  # ratio of model calls left for post finishing
            'dfs_expand': args.dfs_expand,
            'heu': args.use_heu
    }

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(args.device)
    if args.max_len == -1:
        cur_max_len = input_ids.squeeze().size()[0] * 2
        comp_budget = cur_max_len * args.beam_size
    else:
        comp_budget = args.max_len * args.beam_size
        cur_max_len = args.max_len
    
    output = bfs(doc_input_ids=input_ids, model=model, tokenizer=tokenizer, dec_prefix=dec_prefix, avg_score=args.avg_score, max_len=cur_max_len, k_best=args.k_best, comp_budget=comp_budget, config_heu=None, config_search=config_search)

    mo = SearchModelOutput(ends=output)
    print(mo)


if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
