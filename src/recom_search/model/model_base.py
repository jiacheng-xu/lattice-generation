from time import time
from functools import wraps
from abc import ABC, abstractmethod
from transformers import PreTrainedModel, PreTrainedTokenizer


class SearchStrategy(ABC):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device, min_len: int, max_len: int, beam_size: int) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        # self.sos_token_id = sos_token_id
        self.device = device
        self.min_len = min_len
        self.max_len = max_len
        self.beam_size = beam_size

        assert self.model.config.decoder_start_token_id is not None

    # @property
    # @abstractmethod
    # def config(self) -> str:
    #     raise NotImplementedError()


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        result['run_time'] = te-ts
        return result
    return wrap


def cal_model_call(f):
    @wraps(f)
    def wrap(*args, **kw):
        result = f(*args, **kw)
        # t^2
        # result['model_call'] = 0
        return result
    return wrap
