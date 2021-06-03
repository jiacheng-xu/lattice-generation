def load_ctrlsum(device):
    from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast
    model = AutoModelForSeq2SeqLM.from_pretrained("hyunwoongko/ctrlsum-cnndm").to(device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/ctrlsum-cnndm")

    return tokenizer, model

import os, json
import argparse
import torch

def reconstruct_sentence(tokens):
    sent = []
    for tok in tokens:
        ori_text = tok['originalText']
        after = tok['after']
        sent += [ori_text, after]
    return "".join(sent)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="data path containing StanfordNLP docs and tf-idf tokens",
                        default="/mnt/data0/ojas/QFSumm/data/tokenized/eq_100_rescue")
    parser.add_argument("--file_suffix", default='.story.json')
    parser.add_argument("--keyword_suffix", default='.story.json.tfidf-tokens')
    parser.add_argument("--max_nexample", type=int,
                        help="set a max number of examples tested for baseline; -1 means testing on all examples")
    parser.add_argument("--max_inp_sent",default=10,type=int)
    parser.add_argument("--name",default='eq_100_rescue')
    parser.add_argument("--dec_min_len",default=100,type=int)
    parser.add_argument("--dec_max_len",default=150, type=int)
    parser.add_argument("--device",default='cuda:0')
    args = parser.parse_args()
    wt_dir = os.path.join('/mnt/data1/jcxu/qfs_baseline',args.name)
    if not os.path.exists(wt_dir):
        os.mkdir(wt_dir)

    device = args.device

    tokenizer, model = load_ctrlsum(device=device)
    all_files = os.listdir(args.path)
    files_filtered_w_prefix = [ f.split('.')[0] for f in all_files if f.endswith(args.file_suffix)]
    for file_pre in files_filtered_w_prefix:
        #ret files
        raw_file= os.path.join(args.path, f"{file_pre}{args.file_suffix}")
        with open(raw_file,'r') as raw_read_fd:
            raw_doc = json.load(raw_read_fd)
        if args.keyword_suffix != 'none':
            tfidf_token_file= os.path.join(args.path, f"{file_pre}{args.keyword_suffix}")
            with open(tfidf_token_file,'r') as kw_read_fd:
                key_words = eval(kw_read_fd.read())
        
        docId = raw_doc['docId']
        recover_doc = []
        for idx, sent in enumerate(raw_doc['sentences']): 
            tokens = sent['tokens']
            rec_sent  = reconstruct_sentence(tokens)
            if len(rec_sent) > 5: # more than 5 characters
                recover_doc.append((idx, rec_sent))
        recover_doc = recover_doc[:args.max_inp_sent]

        recover_doc_index = [ x[0] for x in recover_doc]
        recover_doc = [ x[1] for x in recover_doc]

        lower_recover_doc = [x.lower() for x in recover_doc]
        text = "".join(recover_doc)[:300*5]
        outputs = []
        if args.keyword_suffix == 'none':
            key_words_in_str = ""
        else:
            key_words_in_str = " | ".join(key_words) + " - "

        data = tokenizer(f"{key_words_in_str}{text}", return_tensors="pt")
        input_ids, attention_mask = data["input_ids"], data["attention_mask"]
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        decoded = model.generate(input_ids, attention_mask=attention_mask, num_beams=5, min_length=args.dec_min_len,max_length=args.dec_max_len)

        output = tokenizer.decode(decoded[0],skip_special_tokens=True)
        print(f"Keyword: {key_words_in_str} Output: {output}")
        outputs.append(output)

        with open(os.path.join(wt_dir, f"{docId}.txt"), 'w') as fd:
            fd.write('\n'.join(outputs))