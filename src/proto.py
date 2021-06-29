from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-xsum-12-6')
tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-xsum-12-6',cache_dir='/Users/jcxu/Code')
BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-xsum-12-6',cache_dir='/Users/jcxu/Code')