
import sys
from src.recom_search.model.topp import GenericSearch
from transformers.testing_utils import require_torch, torch_device
from transformers import is_torch_available
import unittest
# from transformers import logging
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

# from model.topp import GenericSearch

# from recom_search.model.topp import GenericSearch


class GenericSearchTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        sequence_length=10,
        vocab_size=99,
        pad_token_id=0,
        max_length=40,
        num_beams=24,
        length_penalty=2.0,
        do_early_stopping=True,
        num_beam_hyps_to_keep=2,
        temperature=2
    ):
        self.init_model()
        self.parent = parent
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.temperature = temperature
        # cannot be randomely generated
        self.eos_token_id = self.model.config.eos_token_id

    def init_model(self, model_name='facebook/bart-large-cnn'):
        from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
        self.model = BartForConditionalGeneration.from_pretrained(
            model_name).to(torch_device)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        print('Model loaded.', model_name)

    def prepare_inputs(self):
        article = 'The Bart model was proposed in BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer on 29 Oct, 2019.'
        return article

    def check_greedy(self, input_doc):
        gs = GenericSearch(self.model, self.tokenizer,
                           device=torch_device, beam_size=1, do_sample=False, min_len=20, max_len=self.max_length, num_beam_hyps_to_keep=1)
        # https://github.com/huggingface/transformers/blob/e47765d884673e7ee420ed06b4551bfc3d755c8c/src/transformers/generation_utils.py#L943
        output = gs.run(input_doc)
        print(f"Greedy output: {output}")

    def check_beam(self, input_doc):

        gs = GenericSearch(self.model, self.tokenizer,
                           device=torch_device, beam_size=self.num_beams, do_sample=False, min_len=20, max_len=self.max_length, num_beam_hyps_to_keep=self.num_beams//2)

        output = gs.run(input_doc)
        form = '\n'.join(output)
        print(f"BS output: {form}")

    def check_dbs(self, input_doc):
        gs = GenericSearch(self.model, self.tokenizer,
                           device=torch_device, beam_size=self.num_beams, do_sample=False, min_len=20, max_len=self.max_length, num_beam_groups=4, diversity_penalty=1.0, num_beam_hyps_to_keep=self.num_beams)

        output = gs.run(input_doc)
        form = '\n'.join(output)
        print(f"DBS output: {form}")

    def check_topp(self, input_doc):
        gs = GenericSearch(self.model, self.tokenizer,
                           device=torch_device, beam_size=1, do_sample=True, min_len=20, max_len=self.max_length, num_beam_groups=1, num_beam_hyps_to_keep=self.num_beams, top_p=0.7)
        output = gs.run(input_doc)
        form = '\n'.join(output)
        print(f"Topp output: {form}")

    def check_temp(self, input_doc):
        gs = GenericSearch(self.model, self.tokenizer,
                           device=torch_device, beam_size=1, do_sample=True, min_len=20, max_len=self.max_length, num_beam_groups=1, num_beam_hyps_to_keep=self.num_beams, temperature=self.temperature)

        output = gs.run(input_doc)
        form = '\n'.join(output)
        print(f"Temprature output: {form}")


@require_torch
class GenericSearchTest(unittest.TestCase):

    def setUp(self) -> None:
        self.generic_search_tester = GenericSearchTester(self)

    def test_greedy(self):
        inp = self.generic_search_tester.prepare_inputs()
        self.generic_search_tester.check_greedy(inp)

    def test_beam(self):
        inp = self.generic_search_tester.prepare_inputs()
        self.generic_search_tester.check_beam(inp)

    def test_dbs(self):
        inp = self.generic_search_tester.prepare_inputs()
        self.generic_search_tester.check_dbs(inp)

    def test_topp(self):
        inp = self.generic_search_tester.prepare_inputs()
        self.generic_search_tester.check_topp(inp)

    def test_temp(self):
        inp = self.generic_search_tester.prepare_inputs()
        self.generic_search_tester.check_temp(inp)

    def tearDown(self):
        logger.info('Finish test')
