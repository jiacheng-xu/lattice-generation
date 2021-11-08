import statistics
import torch
import logging

from statistics import mode

from typing import List

from src.recom_search.model.setup import tokenizer

debug = False    # fake model output
# debug = True    # fake model output


MODEL_CACHE = '/mnt/data1/jcxu/cache'


def filter_by_score(group: List, top_k=20):
    sorts = sorted(group, key=lambda x: x.get_avg_score(), reverse=True)
    sorts = sorts[:top_k]
    return sorts


def analyze_stat_dict(d):
    result = {}
    for k, v in d.items():
        result[k] = statistics.mean(v)
    return result


def return_str(tokens):
    return tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)



from transformers import MBart50TokenizerFast

# tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE)
# tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
# tokenizer = None


def pnum(num, bit=4):
    if bit == 4:
        return "{:.4f}".format(num)
    else:
        return "{:.2f}".format(num)


def beam_size_policy(beam_size, time_step, policy='regular'):
    if policy == 'regular':
        return beam_size
    else:
        if time_step == 0:
            return beam_size
        else:
            return min(time_step, beam_size)


def render_name(task, data,mname, doc_id, inp_doc_str:str, beam_sz:int, max_len, *args):
    first_few_tokens = inp_doc_str[:20]
    txt = f"{task}_{data}_{doc_id}_{first_few_tokens}_"
    params = [mname, beam_sz, max_len]
    keys = []
    for arg in args:
        for k, v in arg.items():
            params.append(v)
            keys.append(k)
    logging.info("File name: " + "_".join(keys))
    params = "_".join([str(x) for x in params])
    return txt+params



@torch.no_grad()
def run_inference_step(model, input_ids, attention_mask=None, decoder_input_ids=None, targets=None, device='cuda:0', output_dec_hid=False, T=1):
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
                    output_hidden_states=output_dec_hid,
                    use_cache=False, return_dict=True)

    # batch, dec seq, vocab size
    next_token_logits = outputs.logits[:, -1, :]
    if targets is not None:
        targets = targets.to(device)
        loss = torch.nn.functional.cross_entropy(
            input=next_token_logits, target=targets, reduction='none')
    else:
        loss = 0

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
