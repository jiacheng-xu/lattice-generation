
import unittest

from src.recom_search.model.setup import setup_model

class TestSetup(unittest.TestCase):
    def setUp(self):
        from src.recom_search.model.exec_setup import args
        self.args = args
        
    def test_xsum(self):
        self.args.task = 'sum'
        self.args.dataset = 'xsum'
        self.args.hf_model_name = 'facebook/bart-large-xsum'
        tokenizer, model, dataset, dec_prefix = setup_model(self.args.task, self.args.dataset,self.args.hf_model_name,  self.args.device)

    def test_custom_bart(self):
        self.args.task = 'custom'
        self.args.dataset = 'custom_input'
        self.args.hf_model_name = 'facebook/bart-large-xsum'
        tokenizer, model, dataset, dec_prefix = setup_model(self.args.task, self.args.dataset,self.args.hf_model_name,  self.args.device)

    def test_custom_pegasus(self):
        self.args.task = 'custom'
        self.args.dataset = 'custom_input'
        self.args.hf_model_name = 'google/pegasus-xsum'
        tokenizer, model, dataset, dec_prefix = setup_model(self.args.task, self.args.dataset,self.args.hf_model_name,  self.args.device)
    
if __name__ == '__main__':
    unittest.main()