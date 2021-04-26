from util import *

from transformers import BartForConditionalGeneration, BartTokenizer

def write_pkl_to_disk(path: str, fname_prefix: str, data_obj):
    full_fname = os.path.join(path, f"{fname_prefix}.pkl")
    with open(full_fname, 'wb') as fd:
        pickle.dump(data_obj, fd)
    logging.debug(f"Done writing to {full_fname}")


def init_bart_sum_model(mname='sshleifer/distilbart-cnn-6-6', device='cuda:0'):
    model = BartForConditionalGeneration.from_pretrained(mname).to(device)
    tokenizer = BartTokenizer.from_pretrained(mname)
    return model, tokenizer

def bart_decoder_forward_embed(input_ids, embed_tokens, embed_scale):
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    inputs_embeds = embed_tokens(input_ids) * embed_scale
    return inputs_embeds


def summarize_attributions(attributions):
    attributions = attributions.mean(dim=-1)
    attributions = attributions / torch.norm(attributions)
    return attributions

def forward_enc_dec_step(model, encoder_outputs, decoder_inputs_embeds):
    # expanded_batch_idxs = (
    #         torch.arange(batch_size)
    #             .view(-1, 1)
    #             .repeat(1, 1)
    #             .view(-1)
    #             .to(device)
    #     )
    # encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
    #         0, expanded_batch_idxs
    #     )
    model_inputs = {"input_ids": None,
                    "past_key_values": None,
                    "encoder_outputs": encoder_outputs,
                    "decoder_inputs_embeds": decoder_inputs_embeds,
                    }
    outputs = model(**model_inputs, use_cache=False,
                    return_dict=True, output_attentions=True)
    return outputs


def init_bart_family(name_lm, name_sum, device, no_lm=False, no_ood=False):
    if not no_lm:
        lm_model, tok = init_bart_lm_model(name_lm, device)
    else:
        lm_model = None
    sum_model, tok = init_bart_sum_model(name_sum, device)
    if not no_ood:
        if name_sum == "facebook/bart-large-cnn": 
            sum_out_of_domain, _ = init_bart_sum_model(
            "facebook/bart-large-xsum", device)
        else:
            sum_out_of_domain, _ = init_bart_sum_model(
            "facebook/bart-large-cnn", device) 
    else:
        sum_out_of_domain = None
    return lm_model, sum_model, sum_out_of_domain, tok

from captum.attr._utils.visualization import format_word_importances


def simple_viz_attribution(tokenizer, input_ids, attribution_scores):
    token_in_list = input_ids.tolist()
    if isinstance(token_in_list[0], list):
        token_in_list = token_in_list[0]
    words = [tokenizer.decode(x) for x in token_in_list]
    attribution_scores_list = attribution_scores.tolist()
    # for w, ascore in zip(words, attribution_scores_list):
    #     logging.info('{:10} {:02.2f}'.format(w, ascore))

    output = format_word_importances(words, attribution_scores_list)
    return output


@torch.no_grad()
def run_full_model_slim(model, input_ids, attention_mask=None, decoder_input_ids=None, targets=None, device='cuda:0', output_dec_hid=False, output_attentions=False, T=1, special_attn=False):
    decoder_input_ids = decoder_input_ids.to(device)
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    assert decoder_input_ids.size()[0] == input_ids.size()[0]

    model_inputs = {"input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "decoder_input_ids": decoder_input_ids,
                    }

    outputs = model(**model_inputs,
                    output_hidden_states=output_dec_hid, output_attentions=output_attentions,
                    use_cache=False, return_dict=True)

    # batch, dec seq, vocab size
    next_token_logits = outputs.logits[:, -1, :]
    if targets is not None:
        targets = targets.to(device)
        loss = torch.nn.functional.cross_entropy(
            input=next_token_logits, target=targets, reduction='none')
    else:
        loss = 0
    if special_attn:
        cross_attn = outputs['cross_attentions']
        attn = cross_attn[-1][:, :, -1, :]
        # batch, nhead, enc_len
        mean_attn = torch.mean(attn, dim=1)
        # block special positions in input
        mask = (input_ids >= 5).float()
        mean_attn = mean_attn * mask
        return mean_attn[0] 
    if output_attentions:
        # use cross attention as the distribution
        # last layer.   batch=1, head, dec len, enc len
        # by default we use the last layer of attention
        output, p = get_cross_attention(
            outputs['cross_attentions'], input_ids, device=device)
        return output, p

    
    prob = torch.nn.functional.softmax(next_token_logits/T, dim=-1)
    # prob = next_token_logits.softmax(dim=-1)
    next_token = torch.argmax(next_token_logits, dim=-1)
    # next_token = next_token.unsqueeze(-1)
    next_token = next_token.tolist()    # confrim nested list?
    # print(f"Gold: {tokenizer.decode(targets[0].item())}")
    output = [tokenizer.decode(tk) for tk in next_token]
    # logging.info(f"Next token: {output}")
    # outputs['output'] = output
    return output, prob, next_token_logits, loss
