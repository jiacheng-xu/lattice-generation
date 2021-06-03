def load_qa_model_tokenizer(device,mname="bert-large-uncased-whole-word-masking-finetuned-squad"):
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    import torch
    tokenizer = AutoTokenizer.from_pretrained(mname)
    model = AutoModelForQuestionAnswering.from_pretrained(mname).to(device)
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
                        default="/mnt/data0/ojas/QFSumm/data/tokenized/cnndm_small_k1_multi")
    parser.add_argument("--file_suffix", default='.story.json')
    parser.add_argument("--keyword_suffix", default='.story.json.tfidf-tokens')
    parser.add_argument("--max_nexample", type=int,
                        help="set a max number of examples tested for baseline; -1 means testing on all examples")
    parser.add_argument("--max_inp_sent",default=40,type=int)
    parser.add_argument("--name",default='cnndm_small_k1_multi')
    parser.add_argument("--device",default='cuda:0')
    args = parser.parse_args()
    wt_dir = os.path.join('/mnt/data1/jcxu/qfs_baseline',args.name)
    if not os.path.exists(wt_dir):
        os.mkdir(wt_dir)

    device = args.device

    tokenizer, model = load_qa_model_tokenizer(device=device)
    all_files = os.listdir(args.path)
    files_filtered_w_prefix = [ f.split('.')[0] for f in all_files if f.endswith(args.file_suffix)]
    for file_pre in files_filtered_w_prefix:
        #ret files
        raw_file= os.path.join(args.path, f"{file_pre}{args.file_suffix}")
        with open(raw_file,'r') as raw_read_fd:
            raw_doc = json.load(raw_read_fd)
        tfidf_token_file= os.path.join(args.path, f"{file_pre}{args.keyword_suffix}")
        with open(tfidf_token_file,'r') as kw_read_fd:
            key_words = eval(kw_read_fd.read())
        
        docId = raw_doc['docId']
        recover_doc = []
        for idx, sent in enumerate(raw_doc['sentences']): 
            tokens = sent['tokens']
            rec_sent  = reconstruct_sentence(tokens)
            if len(rec_sent)>5:
                recover_doc.append((idx, rec_sent))
        recover_doc = recover_doc[:args.max_inp_sent]

        recover_doc_index = [ x[0] for x in recover_doc]
        recover_doc = [ x[1] for x in recover_doc]

        lower_recover_doc = [x.lower() for x in recover_doc]
        text = "".join(recover_doc)[:300*5]
        extracted_sentences = []
        for key_wd in key_words:
            question = f"What is {key_wd}?"
            inputs = tokenizer(
                question, text, add_special_tokens=True, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"].tolist()[0]
            outputs = model(**inputs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits
            answer_start = torch.argmax(
                answer_start_scores
            )  # Get the most likely beginning of answer with the argmax of the score
            # Get the most likely end of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1
            # answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            answer = tokenizer.decode(input_ids[answer_start:answer_end],skip_special_tokens=True, )
            print(f"Keyword: {key_wd}")
            # print(f"Question: {question}")
            print(f"Answer: {answer}")
            answer = answer.strip()[:20]
            extract_sent_idx = [idx for idx,lower_sent in enumerate(lower_recover_doc) if answer in lower_sent]
            if len(extract_sent_idx) >0:
                sent_idx = extract_sent_idx[0]
                if sent_idx not in extracted_sentences:
                    extracted_sentences.append(sent_idx)

        extracted_sents = [cont for idx,cont in enumerate(recover_doc) if idx in extracted_sentences]
        extracted_sents_idx = [jdx for idx,jdx in enumerate(recover_doc_index) if idx in extracted_sentences]
        # print("\n".join(extracted_sents))
        with open(os.path.join(wt_dir, f"{docId}.txt"), 'w') as fd:
            fd.write(''.join(extracted_sents))
        with open(os.path.join(wt_dir, f"{docId}_idx.txt"), 'w') as fd:
            fd.write(''.join(extracted_sents_idx))