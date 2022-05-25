# [Massive-scale Decoding for Text Generation using Lattices](https://arxiv.org/abs/2112.07660)
[Jiacheng Xu](https://www.cs.utexas.edu/~jcxu/), [Greg Durrett](https://www.cs.utexas.edu/~gdurrett/)

TL;DR: a new search algorithm to construct lattices encoding many generation options; 
two key technical contributions: (1) best-first search, (2) path recombination.



## Visualization
We provide a few examples in the ```vis``` folder and on [my homepage](https://www.cs.utexas.edu/~jcxu/data/summarization/). You need to download the html files to view and **interact** with the model outputs.

The complete set of outputs are available on [Box](https://utexas.box.com/s/wmvhg8lol3kvgirizqyiyiblbn6ogj1a).

## Getting started


- ```model``` contains all of the methods, including baselines like beam search, nucleus sampling, and our methods.
- ```evaluation``` contains scripts for evaluation.
- ```command``` are the prompts and shells we use to run the experiment. 

Beam Search:
```
PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -nexample 100  -ngram_suffix 4  -beam_size 16 -min_len 10 -max_len 35   -model bs 
```

Best-first Search:
```
PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -nexample 100  -ngram_suffix 4  -beam_size 16 -min_len 10 -max_len 35   -model astar_baseline
```

Best-first Search with Recomb:
```
PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -nexample 100  -ngram_suffix 4 -beam_size 16 -min_len 10 -max_len 35 -model astar -merge imp  -avg_score 0.75  -dfs_expand 
```

Best-first Search with Zip:
```
PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -nexample 100  -ngram_suffix 4 -beam_size 16 -min_len 10 -max_len 35 -model astar -merge zip  -avg_score 0.75  -dfs_expand 
```
More detailed instructions coming soon!

## Citation
```
@misc{xu-durrett-2021-massive,
    title={Massive-scale Decoding for Text Generation using Lattices},
    author={Jiacheng Xu and Greg Durrett},
    year={2021},
    eprint={2112.07660},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Contact

jcxu@utexas.edu 