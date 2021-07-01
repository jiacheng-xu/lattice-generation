from numpy.lib.utils import who
from util import *

# a beam search algorithm with merge. 

L = 8  # max diff of two substrings
MAX_STEP = 10 # max gen steps
BS = 10
SUFFIX = 2 # last two tokens match
logging.info(f"BS:{BS} SUFFIX:{SUFFIX} MAX_STEP:{MAX_STEP}")
merge = False
from scipy.stats import entropy
# Each state: UID, log score, previous tokens, top-K and prob for all prev steps

# When to MERGE? When two generation shares some father, and the suffix equals, and the diff is small (len < L)
# global GLOBAL_UID_CNT
GLOBAL_UID_CNT = 0

class Span():
    def __init__(self) -> None:
        # [A: [token, token, token], [score, score, score], B: [token, token], [score, score], .....]
        # prefix if provided
        # we might want something like model repr in future
        pass
import statistics
import math

def compare_ancestor_of_states(bs1, bs2):
    father1 = bs1.get_ancestor_uid()
    father2 = bs2.get_ancestor_uid()
    # trim 
    father1 = father1[:L]
    father2 = father2[:L]
    distance1, distance2 = -1,-1
    for idx, element in enumerate(father1):
        if element in father2:
            distance1 = idx
            break
    if distance1>0:
        distance2 = father2.index(element)
    return distance1, distance2


class BeamState(object):
    def __init__(self, cur_idx_in_distb, cur_distb, cur_distb_token_id,  prev=[]) -> None:
        super().__init__()
        self.score = -math.log(cur_distb[cur_idx_in_distb])
        # self.distb = cur_idx_in_distb # rank in the current peer
        self.token = cur_distb_token_id[cur_idx_in_distb] # token
        self.token_str = tokenizer.decode(self.token) if tokenizer else "[empty]"
        self.prev = prev
        self.assign_uid()
    
    def get_tokens(self):
        tokens = [self.token]
        prev = self.prev
        while prev:
            tokens.append(prev.token)
            prev = prev.prev
        return tokens
    def get_prefix(self):
        tokens = self.get_tokens()[::-1]
        dec_prefix = torch.tensor([tokens],dtype=torch.long).to(device)
        return dec_prefix
    def get_ancestor_uid(self):
        UIDs = []
        prev = self.prev
        while prev:
            UIDs.append(prev.uid)
            prev = prev.prev
        return UIDs

    def extract_prev_score(self):
        scores = []
        prev = self.prev
        while prev:
            scores.append(prev.score)
            prev = prev.prev

        return scores

    def assign_uid(self):
        global GLOBAL_UID_CNT
        self.uid = GLOBAL_UID_CNT
        GLOBAL_UID_CNT += 1

    def get_score(self):
        return statistics.mean(self.extract_prev_score() + [self.score]) 

    def __repr__(self):
        return f"Score: {pnum(self.get_score())}\tTokens: {tokenizer.decode(self.get_tokens()[::-1],clean_up_tokenization_spaces=True)}"

def sublist(lst1, lst2):
    return set(lst1) <= set(lst2)

def fake_model_output(vocab_size=20):
    output = torch.rand(vocab_size) * 20
    softmax_scores =  torch.nn.functional.softmax(output)
    return torch.topk(softmax_scores, k=BS)

# we only do exact suffix matching for now

def merge_compare(beam_a, beam_b):
    # Step 1: suffix match
    a_tokens = beam_a.get_tokens()
    b_tokens = beam_b.get_tokens()
    if len(a_tokens) > SUFFIX and len(b_tokens)> SUFFIX:
        target_tokens = a_tokens[:SUFFIX]
        contain = sublist(target_tokens, b_tokens)
        if a_tokens[:SUFFIX] == b_tokens[:SUFFIX]:
            logging.info(f"Stage 1: SUCCESS")
            pass
        else:
            return None
    else:
        return None


from typing import List

def entrance_merge(beam:List[BeamState]):
    for idx, b in enumerate(beam):
        for jdx in range(idx+1, len(beam)):
            cand_a, cand_b = beam[idx], beam[jdx]
            if (not cand_a) or (not cand_b):
                continue
            merge_compare(cand_a, cand_b)

def run_example(document):
    doc_input_ids = tokenizer(document, return_tensors='pt')['input_ids'][:,:600]
    doc_input_ids = doc_input_ids.to(device)

    whole_beam = [BeamState(cur_idx_in_distb=0,cur_distb=[1.],cur_distb_token_id=[tokenizer.eos_token_id])]
    for t in range(MAX_STEP):
        candidates = []
        for beam_item in whole_beam:
            if not debug:
                # prefix
                decoder_input_ids = beam_item.get_prefix()
                output_tokens, output_prob, output_score, _ = run_full_model_slim(model, doc_input_ids, decoder_input_ids=decoder_input_ids, device=device, output_dec_hid=False, T=1)
                pred_entropy = entropy(output_prob.cpu().numpy(),axis=-1)[0]
                # print(pnum(pred_entropy))
                dynamic_k = min(BS, int(pred_entropy*2))
                # dynamic_k = BS
                values, indices = torch.topk(output_prob, k=dynamic_k)
            else:
                values, indices = fake_model_output()      # replace it with something real
            values = values[0].tolist()
            indices = indices[0].tolist()

            for idx,v,i in zip(range(BS),values,indices):
                tmp_state = BeamState(idx,values,indices,prev=beam_item)
                candidates.append(tmp_state)

        # sort candidates by scores
        sorted_candidates = sorted(candidates, key=lambda x: x.get_score(), reverse=True)
        whole_beam = sorted_candidates[:BS]
    for unit in whole_beam:
        logging.info(unit)
    # print(whole_beam)
    logging.info('done')

if __name__ == '__main__':
    nexample = 20
    cnt = 0
    for example in dataset:
        cnt += 1
        document = example['document']
        sents = document.split('\n')
        inp = "\n".join(sents[:10])

        doc_id = example['id']
        ref_sum = example['summary']
        logging.info(f"\n\n===Inp Doc: {document[:200]}\n---Sum: {ref_sum}")
        run_example(inp)
        if cnt >nexample:
            break
