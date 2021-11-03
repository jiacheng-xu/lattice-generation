from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

MODEL_CACHE = '/mnt/data1/jcxu/cache'
tokenizer =   MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt",cache_dir=MODEL_CACHE)