
import unittest

from src.recom_search.model.baseline import recomb_baseline
import random
random.seed(2021)


def fake_output_machine(t, beam_size):
    assert beam_size >= 2
    if t >= 0:
        values = [0.5, 0.3] + [0.001 for _ in range(beam_size-2)]

        indices = random.sample([1, 2, 3, 4], k=2) + \
            [0 for _ in range(beam_size-2)]
        return values, indices


class BaselineRecomb(unittest.TestCase):
    def setUp(self):
        self.param_sim_function = {
            'ngram_suffix': 2,
            'len_diff': 3
        }
        self.beam_size = 5

    def test_recomb_baseline(self):
        recomb_baseline(doc_input_ids=None, param_sim_function=self.param_sim_function,
                        eos_token_id=10, model=fake_output_machine, debug=True)


if __name__ == '__main__':
    unittest.main()
