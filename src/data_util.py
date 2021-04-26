from util import *
import csv
from attr_util import init_bart_sum_model, run_full_model_slim
from input_grad import step_input_grad
from scipy.stats import entropy


def yield_fact_examples_from_xsum(path_xsum_hall = '/mnt/data1/jcxu/xsum_hallucination_annotations'):
    with open(os.path.join(path_xsum_hall,'hallucination_annotations_xsum_summaries.csv'), 'r') as fd:
        spamreader = csv.reader(fd, delimiter=',')
        all_rows = list(spamreader)
    header = all_rows[0]
    hall_data = all_rows[1:]
    yield hall_data

def yield_fact_examples_from_curated_artifact(path_hall = '/mnt/data1/jcxu/back_to_fact/artifact',file_hall='artifact_hallucination.json'):
    import json
    with open(os.path.join(path_hall,file_hall), 'r') as fd:
        hall_data = json.load(fd)
    yield hall_data

if __name__ == "__main__":
    yielder = yield_fact_examples_from_xsum()
    yielder = yield_fact_examples_from_curated_artifact()
    from datasets import load_dataset
    dataset_xsum = load_dataset('xsum',split='test')
    print("Done loading dataset.")
    device = 'cuda:0'
    model, tokenizer = init_bart_sum_model(device=device,mname='facebook/bart-large-xsum')
    model_pkg = {'sum':model}
    for hall_case in next(yielder):
        bbcid, model_name, sentence, hall_span_content,  start_span, end_span = hall_case[0], hall_case[1], hall_case[2], hall_case[4], int(hall_case[5]), int(hall_case[6])
        if start_span <= 1:
            continue
        example = [ ex for ex in dataset_xsum if ex['id'] == bbcid]
        assert len(example) == 1
        example = example[0]
        document = example['document']
        ref_sum = example['summary']
        doc_input_ids = tokenizer(document, return_tensors='pt')['input_ids'][:,:400]
        logger.info("*"*100)
        logger.info(f"Hallucination span: {hall_span_content}")
        logger.info(f"<strong>Sum: {sentence}</strong>")
        tokens = sentence.split(" ")
        for idx, tok in enumerate(tokens):
            logger.info('-'*10)
            if idx == 0:
                continue
            prefix_tokens = tokens[:idx]
            target_token = tokens[idx]
            tgt_token_id = tokenizer(" " +target_token)['input_ids'][1]

            logger.info(f"<<strong>>Prefix||Target: {' '.join(prefix_tokens)}\t|| {target_token} </strong>")
            prefix_ids = tokenizer(' '.join(prefix_tokens),return_tensors='pt')
            decoder_input_ids = prefix_ids['input_ids'][:,:-1]

            output, prob, next_token_logits, loss = run_full_model_slim(model=model, input_ids=doc_input_ids, decoder_input_ids=decoder_input_ids, device=device)
            # entropy
            squeeze_prob = prob.squeeze()
            show_top_k(squeeze_prob, tokenizer= tokenizer)

            ent_of_pred = entropy(squeeze_prob.cpu().numpy())
            logger.info(f"Entropy: {ent_of_pred: 8.2f}")

            inp_grad_result ,duration = step_input_grad(doc_input_ids.squeeze(), actual_word_id=tgt_token_id, prefix_token_ids=decoder_input_ids, model_pkg=model_pkg, device=device)

        print()