from transformers import PreTrainedModel, PreTrainedTokenizer
from recom_search.model.model_base import SearchStrategy
from recomb_proto import merge_compare
from util import *
from src.recom_search.model.beam_state import BeamState
import heapq
from typing import List
# hash of ngrams



def beam_size_policy(beam_size, time_step, policy='regular'):
    if policy == 'regular':

        return beam_size
    else:
        if time_step == 0:
            return beam_size
        else:
            return min(time_step, beam_size)


class NaiveRecombination(SearchStrategy):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device, min_len: int, max_len: int, beam_size: int) -> None:
        super().__init__(model, tokenizer, device, min_len, max_len, beam_size)
