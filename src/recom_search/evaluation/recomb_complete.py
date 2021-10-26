from collections import defaultdict
import statistics
import torch
from src.recom_search.evaluation.eval_bench import rouge_single_pair
from src.recom_search.model.util import setup_model


with open('3recomb_record.txt', 'r') as fd:
    lines = fd.read().splitlines()


import random
random.shuffle(lines)
lines = lines[:100]

device = torch.device('cuda:0')
print(lines[0])

total_cnt = 0
em = defaultdict(list)
rouges = defaultdict(list)
rouges_l = defaultdict(list)


bucket = [1, 5, 10, 15]
tokenizer, model, dataset = setup_model('cuda:0')
for l in lines:
    inp, seq_a, seq_b = l.split('\t')
    inp = eval(inp)
    seq_a = eval(seq_a)
    seq_b = eval(seq_b)
    len_a, len_b = len(seq_a), len(seq_b)
    inp = torch.LongTensor(inp).to(device).unsqueeze(0)
    seq_a = torch.LongTensor(seq_a).to(device).unsqueeze(0)
    output_a = model.generate(
        input_ids=inp, decoder_input_ids=seq_a).cpu().tolist()[0]
    seq_b = torch.LongTensor(seq_b).to(device).unsqueeze(0)
    output_b = model.generate(
        input_ids=inp, decoder_input_ids=seq_b).cpu().tolist()[0]
    total_cnt += 1
    a_suffix = output_a[len_a:]
    b_suffix = output_b[len_b:]

    print(tokenizer.decode(output_a[:len_a], skip_special_tokens=True),
          "======", tokenizer.decode(a_suffix, skip_special_tokens=True))
    print(tokenizer.decode(output_b[:len_b], skip_special_tokens=True),
          "======", tokenizer.decode(b_suffix, skip_special_tokens=True))
    print('')
    # suffix_len.append(len(a_suffix))
    # suffix_len.append(len(b_suffix))
    # len_diff.append(abs(len(a_suffix) - len(b_suffix)))

    # if a_suffix == b_suffix:
    # em += 1
    for buck in bucket:
        tmp_a_suffix = a_suffix[:buck]
        tmp_b_suffix = b_suffix[:buck]
        if tmp_a_suffix == tmp_b_suffix:
            em[buck].append(1)
        else:
            em[buck].append(0)
        str_a, str_b = tokenizer.decode(tmp_a_suffix, skip_special_tokens=True), tokenizer.decode(
            tmp_b_suffix, skip_special_tokens=True)
        rouge_1_f1 = rouge_single_pair(str_a, str_b)
        rouge_l_f1 = rouge_single_pair(str_a, str_b, 'rougeL')
        rouges[buck].append(rouge_1_f1)
        rouges_l[buck].append(rouge_l_f1)

for buck in bucket:
    r = rouges[buck]
    print(buck, statistics.mean(r), statistics.mean(
        rouges_l[buck]), statistics.mean(em[buck]))


# fd.close()
