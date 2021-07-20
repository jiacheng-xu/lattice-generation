import argparse
from datasets import load_dataset
import statistics
import torch
import logging
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from statistics import mode
import sys
MODEL_CACHE = '/mnt/data1/jcxu/cache'


def setup_logger(name):
    import datetime
    now_time = datetime.datetime.now()
    logname = f"{name}{str(now_time)[:16]}.txt"
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('- %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    file_handler = logging.FileHandler(logname, 'w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)


model_name = 'sshleifer/distilbart-xsum-12-6'
model_name = 'facebook/bart-large-xsum'

tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE)
debug = False    # fake model output
# debug = True    # fake model output
if not debug:
    device = torch.device('cuda:0')
    logging.info('Loading model')
    model = BartForConditionalGeneration.from_pretrained(
        model_name, cache_dir=MODEL_CACHE)
    model = model.to(device)
else:
    device = torch.device('cpu')
logging.info('Loading dataset')
dataset = load_dataset('xsum', split='validation')


def pnum(num):
    return "{:.2f}".format(num)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


@torch.no_grad()
def run_full_model_slim(model, input_ids, attention_mask=None, decoder_input_ids=None, targets=None, device='cuda:0', output_dec_hid=False, T=1):
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
