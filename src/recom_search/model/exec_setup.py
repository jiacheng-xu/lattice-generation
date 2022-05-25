from src.recom_search.model.setup import process_arg,render_address,setup_logger,setup_model

args = process_arg()
dict_io = render_address(root=args.path_output)
setup_logger(name=f"{args.task}_{args.model}_{args.dataset}")
tokenizer, model, dataset, dec_prefix = setup_model(
    task=args.task,dataset= args.dataset, model_name=args.hf_model_name, device_name=args.device)