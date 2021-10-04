
from .model_base import SearchStrategy, timing
import torch
from collections import UserDict
from transformers import PreTrainedModel, PreTrainedTokenizer


class GenericSearch(SearchStrategy):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device, min_len: int, max_len: int, beam_size: int, num_beam_hyps_to_keep: int, temperature: float = 1.0, num_beam_groups: int = 1, do_sample: bool = False, diversity_penalty: float = 0.0, top_p: float = 1.0) -> None:
        super().__init__(model, tokenizer, device, min_len, max_len, beam_size)
        self.num_beam_groups = num_beam_groups
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.do_sample = do_sample
        self.temperature = temperature
        self.diversity_penalty = diversity_penalty
        self.top_p = top_p

    def run(self, input_doc: str):
        input_ids = self.tokenizer(
            input_doc, return_tensors="pt").input_ids.to(self.device)
        run_output = self._timed_run(
            input_ids=input_ids)

        # obtain sequence scores
        scores = run_output['output']['sequences_scores'].cpu().tolist()
        decoded_outputs = self.tokenizer.batch_decode(
            run_output['output']['sequences'], skip_special_tokens=True)
        return decoded_outputs, scores

    @timing
    def _timed_run(self, input_ids):
        outputs = self.model.generate(input_ids=input_ids, top_p=self.top_p, max_length=self.max_len, min_length=self.min_len, num_beams=self.beam_size,
                                      num_beam_groups=self.num_beam_groups,
                                      do_sample=self.do_sample,
                                      num_return_sequences=self.num_beam_hyps_to_keep,
                                      diversity_penalty=self.diversity_penalty,output_scores=True,return_dict_in_generate=True)
        return {'output': outputs}

    @property
    def config(self) -> str:
        return f"Baseline\tp:{self.top_p}  Min/Max:{self.min_len}{self.max_len}"
