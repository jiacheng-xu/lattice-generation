import random
import logging
import random
from typing import List

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

def merge_imp(hash, beam_par, beam_drop):
    """
    We know beam_par and beam_drop have suffix. We are going to merge.
    Basically what we do: make the beam_drop and beam_par the same node.
    """
    pre_node_ids = beam_drop.prev
    assert len(pre_node_ids) == 1
    pre_node_id = pre_node_ids[0]
    pre_node = hash.retrieve_node(pre_node_id)
    if hash.find_root_node_uid(beam_par.uid) == hash.find_root_node_uid(pre_node_id):
        print('Duplicate!')
        return False
    if pre_node in hash.retrieve_node(beam_par.uid).get_antecedent() or beam_par in pre_node.get_antecedent():
        print("loop")
        return False
    # pre_node = hash.retrieve_node(pre_node_id)
    beam_par.prev += [pre_node_id]
    beam_par.prev_score += [beam_drop.score]
    return True

def merge_zip(hash, beam_par, beam_drop, par_match_uids:List[BeamNode]):
    # z = beam_par
    zp = beam_drop
    print(f"Merging {beam_par} | {beam_drop}")
    while zp and par_match_uids:
        par_node_uid = par_match_uids.pop(0)
        par_node = hash.retrieve_node(par_node_uid)
        assert par_node.token_idx == zp.token_idx
        logging.debug(f"Replacing {zp.token_str} with {par_node.token_str}")
        if hash.find_root_node_uid(par_node_uid) == hash.find_root_node_uid(zp.uid)   :
            logging.debug('Duplicate!')
            return
        if par_node in hash.retrieve_node(zp.uid).get_antecedent() or zp in par_node.get_antecedent():
            logging.debug("loop")
            return 
        hash.replace_node(par_node_uid, zp.uid)
        par_node.prev += zp.prev
        par_node.prev_score += zp.prev_score
        next_zp_id = zp.prev[0]
        zp = hash.retrieve_node(next_zp_id)
    # print('----------')
    return 
