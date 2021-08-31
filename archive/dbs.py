
from collections import UserDict
import torch


from argparse import ArgumentParser

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    HammingDiversityLogitsProcessor,
    BeamSearchScorer,
)


from .model_base import SearchStrategy, timing


class DBS(SearchStrategy):

    def __init__(self, model, tokenizer, device, min_len: int, max_len: int, beam_size: int, beam_group: int, num_beam_hyps_to_keep: int, hamming_penalty: float = 1.0) -> None:
        super().__init__(model, tokenizer, device, min_len, max_len, beam_size)
        self.beam_group = beam_group
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.hamming_penalty = hamming_penalty
        self.beam_scorer = BeamSearchScorer(
            batch_size=1,
            max_length=self.max_len,
            num_beams=self.beam_size,
            device=self.device,
            num_beam_groups=self.beam_group,
            num_beam_hyps_to_keep=self.num_beam_hyps_to_keep
        )
        self.logits_processor = LogitsProcessorList([
            HammingDiversityLogitsProcessor(
                self.hamming_penalty, num_beams=self.beam_size, num_beam_groups=self.beam_group),
            MinLengthLogitsProcessor(
                self.min_len, eos_token_id=self.model.config.eos_token_id),
        ])

    def run(self, input_doc: str):
        encoder_input_ids = self.tokenizer(
            input_doc, return_tensors="pt").input_ids.to(self.device)
        input_ids = torch.ones((self.beam_size, 1),
                               device=self.device, dtype=torch.long)
        input_ids = input_ids * self.model.config.decoder_start_token_id
        model_kwargs = {"encoder_outputs": self.model.get_encoder()(
            encoder_input_ids.repeat_interleave(self.beam_size, dim=0), return_dict=True)}
        run_output = self._timed_run(
            input_ids=input_ids, model_kwargs=model_kwargs)
        decoded_outputs = self.tokenizer.batch_decode(
            run_output.output, skip_special_tokens=True)

    @timing
    def _timed_run(self, input_ids, model_kwargs):
        outputs = self.model.group_beam_search(
            input_ids, self.beam_scorer, logits_processor=self.logits_processor, max_length=self.max_len, **model_kwargs)
        return UserDict({'output': outputs})

    @property
    def config(self) -> str:
        return f"DBS\tGroup:{self.beam_group} Pen:{self.hamming_penalty} Min/Max:{self.min_len}{self.max_len}"
