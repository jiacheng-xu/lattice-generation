import random
import logging
import random
from re import A

from transformers.utils.dummy_pt_objects import BertForMaskedLM
from src.recom_search.model.bfs_util import HashedGen
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
    pass

def merge_zip(hash, beam_par, beam_drop):
    z = beam_par
    zp = beam_drop
    while len(zp.prev)
    pass