from util import *

# based on current prediction, predict future k steps
def future_simulation(model, device,input_doc_token_ids, prefix_token_ids, max_expand_steps=5,min_expand_prob=0.1):
    logger.info(f"Simulation Prefix: {tokenizer.decode(prefix_token_ids,skip_special_tokens=True)}")
    for t in range(max_expand_steps):
        pass
    


def load_xsum(split='validation'):
    from datasets import load_dataset
    dataset_xsum = load_dataset('xsum',split=split)
    return dataset_xsum
    
if __name__ == '__main__':
    
    pass
