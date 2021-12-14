import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
from rouge_score import rouge_scorer
from src.recom_search.evaluation.eval_bench import rouge_single_pair


from src.recom_search.model.model_output import SearchModelOutput
name_dir = '/mnt/data1/jcxu/lattice-sum/output/data/sum_xsum_bs_16_35_False_0.4_False_False_4_5_zip_-1_0.0_0.9'
# name_dir = '/mnt/data1/jcxu/lattice-sum/output/data/sum_xsum_bs_64_35_False_0.4_False_False_4_5_zip_-1_0.0_0.9'
files = os.listdir(name_dir)
all_dps = []
for f in files:
    with open(os.path.join(name_dir, f), 'rb') as fd:
        data: SearchModelOutput = pickle.load(fd)
    texts = data.output
    avg_scores = data.score_avg
    sum_scores = data.score
    ref = data.reference
    r2s = [rouge_single_pair(text, ref, 'rouge2') * 100 for text in texts]

    highest_score = max(sum_scores)
    # highest = max(r2s)  # who has the highest ROUGE score
    idx_high_model = [idx for idx, modelscore in enumerate(
        sum_scores) if modelscore == highest_score][0]
    dp = [(r2s[idx_high_model]-r2, sum_scores[idx_high_model] - sscore)
          for r2, sscore in zip(r2s, sum_scores)]
    all_dps += dp
print(all_dps)

df = pd.DataFrame(all_dps, columns=['deltaR2', 'deltaScore'])
df.to_pickle("bs16.pkl")
sns.set_theme(style="white", color_codes=True)
plt.rcParams["figure.figsize"] = (5, 2.2)

# fig, ax = plt.subplot()

fig, ax = plt.subplots()

# ax.scatter(x=df['deltaR2'], y=df['deltaScore'],alpha=0.2)
ax = sns.regplot(x=df['deltaR2'], y=df['deltaScore'], marker=".", line_kws={
                 'color': 'green'}, scatter_kws={'alpha': 0.2})
ax.grid(True)


print(df['deltaR2'].corr(df['deltaScore'], method='pearson'))
ax.set_xlabel(r'$R2(h^{*}) - R2(h)$')
# ax.set_xlabel(r'\textbf{time (s)}')
ax.set_ylabel(r'$s(h^{*}) - s(h)$')
fig.tight_layout()
# plt.show()
print(os.getcwd())
plt.savefig('rouge_score_cor.pdf')
print(df['deltaR2'].corr(df['deltaScore'], method='pearson'))
# print(df['delta_rouge'].corr(df['delta_score_top'],method='pearson'))
