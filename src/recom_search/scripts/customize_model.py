"""
Import the model you would like to use in this file. 
Tips:
1. You can use model_name.from_pretrained(model_name_example, cache_dir=xxx) to cache the model and tokenizer. 
2. 
"""


"""
MT Example
"""
dataset_name = "zh-en"
# MT dataset should be like "xx-en" or "xx-yy"
customize_model_name = "facebook/mbart-large-50-many-to-one-mmt"

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
config = AutoConfig.from_pretrained(customize_model_name)
model = AutoModelForSeq2SeqLM.from_config(config)
tokenizer = AutoTokenizer.from_config(config)

from transformers.models.mbart.tokenization_mbart import FAIRSEQ_LANGUAGE_CODES


assert dataset_name.endswith('-en')
src_lang = dataset_name[:2]
match = [x for x in FAIRSEQ_LANGUAGE_CODES if x.startswith(src_lang)]
assert len(match) == 1
lang = match[0]
tokenizer.src_lang = lang
dataset = read_mt_data(name=dataset)
dec_prefix = [tokenizer.eos_token_id, tokenizer.lang_code_to_id["en_XX"]]
logging.info(f"{tokenizer.decode(dec_prefix)}")