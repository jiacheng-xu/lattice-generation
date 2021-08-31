Evaluation Metrics

Text quality analysis
- Readability/Simplicity: textstat https://github.com/ShuyangCao/inference_style_control/blob/116d96e5f6e1f436adbc7321bf92e4e3f2e258fa/evaluation/eval_readability.py
- Content overlap: noun_chunks spacy
- Overlap: self-ROUGE, ngram
- Reference match: ROUGE, BERTScore

Model analysis:
- Count of model call: breakdown including simulation calls, runing calls
