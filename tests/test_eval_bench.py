
import sys
from src.recom_search.evaluation.eval_bench import *
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


@require_torch
class EvalBenchTest(unittest.TestCase):

    def setUp(self) -> None:
        self.inp1 = ["Today is a good day for everyone.", "Today is a good day for everyone."]
        self.inp2 = ["Today is a good day for everyone.", "The dog sits on the mat."]
