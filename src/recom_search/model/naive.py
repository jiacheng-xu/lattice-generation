from transformers import PreTrainedModel, PreTrainedTokenizer
from util import *
import heapq
from typing import List


def beam_size_policy(beam_size, time_step, policy='regular'):
    if policy == 'regular':
        return beam_size
    else:
        if time_step == 0:
            return beam_size
        else:
            return min(time_step, beam_size)

