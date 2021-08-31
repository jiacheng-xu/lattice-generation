from transformers import PreTrainedModel, PreTrainedTokenizer
from src.recom_search.model.model_base import SearchStrategy

from src.recom_search.model.recomb_proto import merge_compare
from .util import *
from src.recom_search.model.beam_state import BeamState

import heapq

class BestFirstRecombination(SearchStrategy):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device, min_len: int, max_len: int, beam_size: int) -> None:
        super().__init__(model, tokenizer, device, min_len, max_len, beam_size)
    
def greedy_generate_sequence(doc_input_ids, model, start_seed, device, max_len=20, extra_steps=7):
    pointer = start_seed
    decoder_input_ids = pointer.get_tokens_as_input()
    cur_len = pointer.len
    if extra_steps > 0:
        target = cur_len + extra_steps
    else:
        target = max_len

    fish_bone_teacher = []
    fish_bone_student = []

    while cur_len < target:
        decoder_input_ids = pointer.get_tokens_as_input()
        output_tokens, output_prob, output_score, _ = run_full_model_slim(
            model, doc_input_ids, decoder_input_ids=decoder_input_ids, device=device, output_dec_hid=False, T=1)
        values, indices = torch.topk(output_prob, k=2)
        values = values[0].tolist()
        indices = indices[0].tolist()
        next_state = BeamState(0, values, indices, prev=pointer)
        store_state = BeamState(1, values, indices, prev=pointer)

        pointer = next_state
        cur_len = pointer.len
        if pointer.finished:
            break
        if store_state.finished:
            continue
        fish_bone_teacher.append(next_state)
        fish_bone_student.append(store_state)
    return pointer, fish_bone_teacher, fish_bone_student

def rollout_recombine(doc_input_ids, model, seed:BeamState, teacher_seed:BeamState,teacher_final_seed,  extra_steps=10, max_len=20):
    # try to rollout the seed, check if it can be recombined
    # if yes, update the teacher seed,
    # if no, finish generation and return the new student seed
    prefix_len = seed.len - 1
    teacher_len = teacher_final_seed.len
    # teacher seed xxxxx, yyyyy[y]
    # seed: xxxxx, z[7 steps]?
    roll_out_state, fb_teacher, fb_student = greedy_generate_sequence(doc_input_ids, model,
                                                        start_seed=seed, device=device, extra_steps=min(extra_steps, max_len - prefix_len))# try to gen and merge
    # target states
    data = []
    cur = teacher_final_seed
    for _ in range(teacher_len - prefix_len):
        data.append(cur)
        cur = cur.prev
    data = data[::-1]
    flag_recomb = False
    for d in data:
        output_a, output_b = merge_compare(d, roll_out_state,merge_to_a=True)
        if output_a == None or output_b == None:
            flag_recomb = True
            # print()
            output = output_a or output_b
            break
    if flag_recomb == False:
        output = [roll_out_state, fb_teacher, fb_student]
    
    return flag_recomb, output
    
def random_teacher(max_len, eos_token_id, pad_token_id):
    cnt = 0
    start = BeamState(cur_idx_in_distb=0, prob_distrib=[1., 0, 0, 0, 0], token_id_distb=[
        eos_token_id, pad_token_id, pad_token_id, pad_token_id, pad_token_id])
    pointer = start
    while cnt < max_len:
        another = BeamState(cur_idx_in_distb=0, prob_distrib=[1., 0, 0, 0, 0], token_id_distb=[
        eos_token_id, pad_token_id, pad_token_id, pad_token_id, pad_token_id], prev=pointer)
        pointer = another
        cnt += 1
    return pointer

def best_first_search(doc_input_ids, model,  pad_token_id=0, eos_token_id=21, max_len=20, explore_cnt=100):
    extra_steps = 10
    init_seed = BeamState(cur_idx_in_distb=0, prob_distrib=[1., 0, 0, 0, 0], token_id_distb=[
        eos_token_id, pad_token_id, pad_token_id, pad_token_id, pad_token_id])

    # construct init teacher
    teahcer = random_teacher(max_len,eos_token_id,pad_token_id)
    h = []
    heapq.heappush(h, (-init_seed.prob, init_seed, teahcer, teahcer))
    outputs = []
    recombine_outputs = []
    explored_cnt = 0
    while h:
        s = heapq.heappop(h)
        explored_cnt += 1
        score, seed, teacher_seed, last_state = s
        flag, rollout_output = rollout_recombine(doc_input_ids,model, seed,teacher_seed,last_state, extra_steps,max_len)
        # If merge successful, merge and move on
        # If not, finish the generation and store
        if flag:
            recombine_outputs.append(rollout_output)
        else:
            roll_out_state, fb_teacher, fb_student = rollout_output
            final_roll_out_state, rest_fb_teacher, rest_fb_student = greedy_generate_sequence(doc_input_ids, model,
                                                        start_seed=roll_out_state, device=device, extra_steps=-1, max_len=max_len)
            
            for state_teacher, state_student in zip(fb_teacher, fb_student):
                heapq.heappush(h, (-state_student.prob, state_student,state_teacher, final_roll_out_state))
            for state_teacher, state_student in zip(rest_fb_teacher, rest_fb_student):
                heapq.heappush(h, (-state_student.prob, state_student,state_teacher, final_roll_out_state))
            outputs.append(final_roll_out_state)
            final_tokens = final_roll_out_state.get_tokens()
            prefix_len = seed.len
            logging.info(f"=======>{return_str(final_tokens[:prefix_len-1])} ||| {return_str(final_tokens[prefix_len-1:])}\t\t  -{pnum(score)}")
            # logging.info(f"From: {seed.get_output_str()}\nExecute: {final_roll_out_state.get_output_str()}")

        # print('Done')
        
        if explored_cnt >= explore_cnt:
            break
    logging.info(f"*************Regular len: {len(outputs)}*************")
    logging.info(f"*************Recomb len: {len(recombine_outputs)}*************")
    # UIDs of recombs
    recom_uids = [ x.uid for x in recombine_outputs]
    outputs = [x for x in outputs if x.uid not in recom_uids]
    logging.info('\n\nStart of recom')
    for rec in recombine_outputs:
        logging.info(repr(rec))
    logging.info('\n\nStart of regular')
    for regular in outputs:
        logging.info(repr(regular))
    return None
