from datasets import load_dataset

dataset = load_dataset('xsum', 'test')
d = dataset['validation']
cnt = 100
for ex in d:
    print(ex['summary'])
    cnt -= 1
    if cnt<=0:
        break