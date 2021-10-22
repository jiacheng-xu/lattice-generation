import logging
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


def core_merge(beam_par: BeamNode, beam_drop: BeamNode, doc_input_ids=None):
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
    while pointer_drop and par_paths:
        if pointer_drop in par_paths:
            return # merge fail
        next_par_paths = []
        for par_path in par_paths:
            if pointer_drop.token_idx == par_path.token_idx:
                next_par_paths += par_path.prev
        if next_par_paths:
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
    for path in prev_par_paths:
        path.add_prev_node(pointer_drop, score)
    beam_par.print_lattice()
    return beam_par
    # go leftward to end of prev_par_paths, get all nodes
    # go leftward to end of
