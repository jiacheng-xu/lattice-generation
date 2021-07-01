from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-xsum")

from transformers.generation_utils import GenerationMixin
from transformers.generation_utils import top_k_top_p_filtering

ARTICLE_TO_SUMMARIZE = "The country's consumer watchdog has taken Apple to court for false advertising because the tablet computer does not work on Australia's 4G network. Apple's lawyers said they were willing to publish a "

inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024,  return_tensors='pt')

# Generate Summary
summary_ids = model.generate(inputs['input_ids'], top_k=10000, num_beams=4, max_length=30, early_stopping=True)
print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])