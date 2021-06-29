from attr_util import write_pkl_to_disk,init_bart_sum_model, init_bart_family
from transformers.modeling_outputs import BaseModelOutput

from util import *
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
from attr_util import forward_enc_dec_step, bart_decoder_forward_embed, summarize_attributions, simple_viz_attribution




def step_input_grad(input_ids, actual_word_id, prefix_token_ids, model_pkg, device):
    start_time = time.time()
    input_ids = torch.LongTensor(input_ids).to(device).unsqueeze(0)
    # input_ids = input_ids[:, :400]
    batch_size, seq_len = input_ids.size()
    assert batch_size == 1
    # encode enc input
    model_encoder = model_pkg['sum'].model.encoder

    # encode dec input
    decoder_input_ids = prefix_token_ids.to(device)
    # decoder_input_ids = prefix_token_ids.repeat((num_run_cut, 1))

    dec_seq_len = decoder_input_ids.size()[-1]
    model_decoder = model_pkg['sum'].model.decoder
    embed_scale = model_decoder.embed_scale
    embed_tokens = model_decoder.embed_tokens
    # dec input embedding
    dec_inp_embedding = bart_decoder_forward_embed(
        decoder_input_ids, embed_tokens, embed_scale)

    with torch.enable_grad():
        encoder_outputs = model_encoder(input_ids.to(device), return_dict=True)
        encoder_outputs.last_hidden_state.retain_grad()

        interp_out = forward_enc_dec_step(
            model_pkg['sum'], encoder_outputs=encoder_outputs, decoder_inputs_embeds=dec_inp_embedding)

        logits = interp_out.logits[:, -1, :]
        target = torch.LongTensor([actual_word_id]).to(device)

        loss = torch.nn.functional.cross_entropy(logits, target)

        loss.backward()
        logger.info(f"Loss: {loss.tolist()}")
        raw_grad = encoder_outputs.last_hidden_state.grad
        # print(raw_grad.size())  # 1, 563, 1024
        # print(encoder_outputs.last_hidden_state.size())
        result_inp_grad = raw_grad * encoder_outputs.last_hidden_state
        # result_inp_grad = torch.mean(result_inp_grad, dim=-1)

    ig_enc_result = summarize_attributions(result_inp_grad)
    duration = time.time() - start_time
    # ig_dec_result = summarize_attributions(ig_dec_result)
    if random.random() < 0.99:
        extracted_attribution = ig_enc_result.squeeze(0)
        input_doc = input_ids.squeeze(0)
        viz = simple_viz_attribution(
            tokenizer, input_doc, extracted_attribution)
        logger.info(viz)
        # extracted_attribution = ig_dec_result
        # viz = simple_viz_attribution(
        #     tokenizer, decoder_input_ids[0], extracted_attribution)
        # logger.info(viz)
    return ig_enc_result, duration


if __name__ == "__main__":
    parser = common_args()

    args = parser.parse_args()
    args = fix_args(args)
    logger.info(args)

    device = args.device

    model_lm, model_sum, model_sum_ood, tokenizer = init_bart_family(
        args.mname_lm, args.mname_sum, device, no_lm=True, no_ood=True)
    logger.info("Done loading BARTs.")
    model_pkg = {'sum': model_sum, 'tok': tokenizer}
    all_files = os.listdir(args.dir_base)
    for f in all_files:
        outputs = []
        step_data, meta_data = read_meta_data(args.dir_meta, f)
        uid = meta_data['id']
        sent_token_ids = meta_data['sent_token_ids']
        output_base_data = load_pickle(args.dir_base, f)
        acc_duration = 0
        for t, step in enumerate(step_data):
            output_base_step = output_base_data[t]
            if args.sent_pre_sel:
                input_doc = prepare_filtered_input_document(
                    output_base_step, sent_token_ids)
            else:
                input_doc = meta_data['doc_token_ids'][:args.hard_max_len]
        
            inp_grad_result ,duration= step_input_grad(input_doc, actual_word_id=step['tgt_token_id'], prefix_token_ids=step['prefix_token_ids'], model_pkg=model_pkg, device=device)
            acc_duration += duration
            result = inp_grad_result.squeeze(0).cpu().detach()
            if args.sent_pre_sel:
                rt_step = {
                    'doc_token_ids': input_doc,
                    'output': result
                }
                outputs.append(rt_step)
            else:
                outputs.append(result)

        skinny_meta = {
            'doc_token_ids': meta_data['doc_token_ids'],
            'map_index':meta_data['map_index'],
            'sent_token_ids':meta_data['sent_token_ids'],
            'output': outputs,
            'time': acc_duration
        }
        write_pkl_to_disk(args.dir_task, uid, skinny_meta)
        print(f"Done {uid}.pkl")