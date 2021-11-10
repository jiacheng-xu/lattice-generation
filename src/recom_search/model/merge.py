
import random
import logging
import random

from transformers.utils.dummy_pt_objects import BertForMaskedLM

from src.recom_search.model.beam_state import BeamNode


def similarity_heuristic(a_tokens, b_tokens, ngram_suffix, len_diff) -> bool:

    if len(a_tokens) > ngram_suffix and len(b_tokens) > ngram_suffix:
        if a_tokens[-ngram_suffix:] == b_tokens[-ngram_suffix:]:
            # logging.debug(f"Stage 1: Suffix match SUCCESS")
            pass
        else:
            return False
    else:
        return False

    # Stage 2: length
    if abs(len(a_tokens) - len(b_tokens)) < len_diff:
        logging.debug(f"Stage 2: Len Diff SUCCESS")
    else:
        # logging.debug(f"Stage 2: Len Diff FAIL")
        return False
    return True


def write_recomb_records(match_suffix, seq_a, seq_b, doc_input, ngram_suffix, save_file='recomb_record.txt'):
    list_doc_input = doc_input.squeeze().cpu().tolist()
    prefix_a_token = seq_a.all_token_idx
    prefix_b_token = seq_b.all_token_idx
    a = prefix_a_token + match_suffix
    b = prefix_b_token + match_suffix
    with open(str(ngram_suffix)+save_file, 'a') as fd:
        fd.write(f"{list_doc_input}\t{a}\t{b}\n")


def new_core_merge(beam_par: BeamNode, beam_drop: BeamNode, hash= None, doc_input_ids=None, ngram_suffix=None):
    pointer_par = beam_par
    pointer_drop = beam_drop

    group_par = [(None, 0, pointer_par)]
    group_drop = [(None, 0, pointer_drop)]
    """
    # par: [prefix=4,3, cur=7], [prefix=4,3, cur=2]
    # son: [prefix=4,3, cur=7], [prefix=4,3, cur=8], [prefix=4,3, cur=11]
    # if son's node_i can't find a match in par's nodes, add it the prev list of last matched par node
    # if son's node_i can find, update hash, move on.
    """
    ngram_cnt = 0
    while group_par and group_drop:
        matched_drop = []
        matched_par = []
        unmatched_drop = []
        for drop in group_drop:
            drop_prefix, drop_score, drop_node = drop
            # try to match my self to some in parent
            match = None
            for i in range(len(group_par)):
                tmp_par = group_par[i]
                if tmp_par == None:
                    continue
                par_prefix, par_score, par_node = tmp_par
                if drop_prefix == par_prefix and drop_node.token_idx == par_node.token_idx:
                    match = tmp_par
                    group_par[i] = None
                    break
            if match:
                matched_drop.append(drop)
                matched_par.append(match)
            else:
                unmatched_drop.append(drop)


        # some match, some unmatch
        next_group_par = []
        next_group_drop = []
        for x, y in zip(matched_par, matched_drop):
            prefix_x, prefix_score_x, node_x = x
            prefix_y, prefix_score_y,  node_y = y
            # everything in cache end and value = node_y will be renamed to node_x
            if hash is not None:
                hash.dead_id.append(node_y.uid)

            new_prefix = node_x

            next_node_par = node_x.prev    # List
            next_node_par_score = node_x.prev_score    # List
            for _node, _score in zip(next_node_par, next_node_par_score):
                next_group_par.append((new_prefix, _score, _node))

            next_node_drop = node_y.prev
            next_node_drop_score = node_y.prev_score
            for _node, _score in zip(next_node_drop, next_node_drop_score):
                next_group_drop.append((new_prefix,  _score, _node))

        for z in unmatched_drop:
            prefix, score,  node = z
            prefix.add_prev_node(node, score)
        # all unmatch
        #
        group_par = next_group_par
        group_drop = next_group_drop
        ngram_cnt += 1
        if ngram_suffix!= None and  ngram_cnt >= ngram_suffix:
            break
    if ngram_suffix != None and group_drop and group_par:
        for drop in group_drop:
            drop_prefix, drop_score, drop_node = drop
            # try to match my self to some in parent
            match = None
            for i in range(len(group_par)):
                tmp_par = group_par[i]
                if tmp_par == None:
                    continue
                par_prefix, par_score, par_node = tmp_par
                if drop_prefix == par_prefix:
                    # if drop_prefix == par_prefix and drop_node.token_idx == par_node.token_idx:
                    par_prefix.add_prev_node(drop_node, drop_score)
                    group_par[i] = None
                    break

    return pointer_par


def core_merge(beam_par: BeamNode, beam_drop: BeamNode, doc_input_ids=None, ngram_suffix=None):
    """
    beam_par is the node to keep, beam_drop is to "discard"
    our goal is to merge them into a larger lattice ending with beam_par.uid
    """
    # logging.debug(beam_par.all_token_idx)
    # logging.debug(beam_drop.all_token_idx)

    # when does their suffix starts to differ?
    pointer_par = beam_par
    pointer_drop = beam_drop
    # we just assume they share a same suffix
    par_paths = [pointer_par]
    # beam_drop is treated as a single line
    prev_par_paths = par_paths
    prev_pointer_drop = beam_drop
    matched_suffix = []
    while pointer_drop and par_paths:
        if pointer_drop in par_paths:
            return  # merge fail
        next_par_paths = []
        for par_path in par_paths:
            if pointer_drop.token_idx == par_path.token_idx:

                next_par_paths += par_path.prev
        if next_par_paths:
            matched_suffix.append(pointer_drop.token_idx)
            prev_pointer_drop = pointer_drop
            pointer_drop = pointer_drop.prev[-1]
            prev_par_paths = par_paths
            par_paths = next_par_paths
        else:
            break    # suffix match end
    # pointer_drop is the first token that differs
    # par_paths is the first threads differs
    # prev_par_paths is the last match
    # add pointer_drop to prev_par_paths 's prev

    # the score of pointer_drop -> prev_pointer_drop
    score = prev_pointer_drop.score

    for path in par_paths:
        if doc_input_ids is not None:
            write_recomb_records(
                matched_suffix[::-1], path, pointer_drop, doc_input=doc_input_ids, ngram_suffix=ngram_suffix)
    for path in prev_par_paths:

        # print(pointer_drop)
        path.add_prev_node(pointer_drop, score)
    # beam_par.print_lattice()
    return beam_par


    # go leftward to end of prev_par_paths, get all nodes
    # go leftward to end of
if __name__ == '__main__':
    ng = 2
    hash = HashedGen(ng)
    n1 = BeamNode(1, 1, [], [random.random()])
    n2 = BeamNode(1, 2, [n1], [random.random()])
    n6 = BeamNode(1, 6, [n2], [random.random()])
    n7 = BeamNode(1, 7, [n6], [random.random()])
    n3 = BeamNode(1, 3, [n7, n2], [random.random(), random.random()])
    n4 = BeamNode(1, 4, [n3], [random.random()])
    n23 = BeamNode(1, 23, [n4], [random.random()])
    n24 = BeamNode(1, 24, [n23], [random.random()])

    n11 = BeamNode(1, 11, [n1], [random.random()])
    n9 = BeamNode(1, 9, [n11], [random.random()])
    m7 = BeamNode(1, 7, [n9], [random.random()])
    n8 = BeamNode(1, 8, [n11], [random.random()])
    m3 = BeamNode(1, 3, [m7, n11, n8], [random.random(),
                                        random.random(), random.random()])
    m4 = BeamNode(1, 4, [m3], [random.random()])
    m23 = BeamNode(1, 23, [m4], [random.random()])
    m24 = BeamNode(1, 24, [m23], [random.random()])
    hash.add_helper(n1, n2)
    hash.add_helper(n2, n6)
    hash.add_helper(n6, n7)
    hash.add_helper(n7, n3)
    hash.add_helper(n3, n4)
    hash.add_helper(n2, n3)
    hash.add_helper(n4, n23)
    hash.add_helper(n23, n24)
    hash.add_helper(n1, n11)
    hash.add_helper(n11, n9)
    hash.add_helper(n9, m7)
    hash.add_helper(m7, m3)
    hash.add_helper(n11, m3)
    hash.add_helper(m3, m4)
    hash.add_helper(m4, m23)
    hash.add_helper(m23,m24)

    output_node = new_core_merge(n24, m24, hash,ngram_suffix=ng)
    # output_node.print_lattice()
    print(hash.data)
