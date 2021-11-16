
import random
import logging
import random


from src.recom_search.model.beam_state import BeamNode,BeamNodeEz


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
    
    pointer_par.add_prev_node(pointer_drop.prev[0], pointer_drop.score)

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
