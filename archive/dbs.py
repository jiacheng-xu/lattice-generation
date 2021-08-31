from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    HammingDiversityLogitsProcessor,
    BeamSearchScorer,
)
import torch

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
device = torch.device('cuda:0')
model = model.to(device)
encoder_input_str = "summarize: Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts every language problem into a text-to-text format. Our systematic study compares pretraining objectives, architectures, unlabeled datasets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our dataset, pre-trained models, and code."
encoder_input_str = "summarize: After playing coy on the subject, GOP House Minority Leader Kevin McCarthy is planning to appoint Republicans to the select committee to investigate the Jan. 6 attack on the Capitol, Republican sources familiar with his plans tell ABC News. House Speaker Nancy Pelosi announced last week that Democrats would move forward with creating the select committee after Senate Republicans blocked a proposal for an independent, bipartisan commission.McCarthy -- who will get five appointments to the committee -- hadn't initially decided whether he would appoint anyone at all and reportedly privately threatened Republicans who would accept an appointment by Pelosi."
print(encoder_input_str)
encoder_input_ids = tokenizer(
    encoder_input_str, return_tensors="pt").input_ids.to(device)


# lets run diverse beam search using 6 beams
num_beams = 50
ngroups = 10
# define decoder start token ids
input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
input_ids = input_ids * model.config.decoder_start_token_id

# add encoder_outputs to model keyword arguments
model_kwargs = {
    "encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
}
val_pen = [0.001, 0.5, 1., 5., 10.]
for penalty in val_pen:
    beam_scorer = BeamSearchScorer(
        batch_size=1,
        max_length=model.config.max_length,
        num_beams=num_beams,
        device=model.device,
        num_beam_groups=ngroups,
        num_beam_hyps_to_keep=20
    )

    # instantiate logits processors
    logits_processor = LogitsProcessorList([
        HammingDiversityLogitsProcessor(penalty, num_beams=num_beams, num_beam_groups=ngroups),
        MinLengthLogitsProcessor(20, eos_token_id=model.config.eos_token_id),
    ])

    outputs = model.group_beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f"---------------{penalty}---------------")
    for do in decoded_outputs:
        print(do)