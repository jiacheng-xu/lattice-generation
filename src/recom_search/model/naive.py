from transformers import PreTrainedModel, PreTrainedTokenizer
from recom_search.model.model_base import SearchStrategy
from recomb_proto import merge_compare
from util import *
from recomb_data_struct import BeamState
import heapq

class NaiveRecombination(SearchStrategy):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device, min_len: int, max_len: int, beam_size: int) -> None:
        super().__init__(model, tokenizer, device, min_len, max_len, beam_size)
