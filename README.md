# back_to_fact
Two major components here:
1. path recombination
2. dynamic beam expansion

Metrics:
1. ROUGE
2. Grammarticality
3. Faithfulness
4. Diversity
5. Query-focused setting (control)

## Path Recombination
Keywords: structual, lattice, eliciting LM, 

Basic case: a long and nice path

For time step t, the model predicts a 

For time step t, $p(w_{t})=q$, and after N time steps, p(w_{t+N})=v


Cases to consider:

A. a 55-year-old man, 55, goes ...

B. one good effect and one bad effect

C. 

## Dynamic Beam Expansion


For time step t, p(w_{t})=q, and after N time steps, p(w_{t+N})=v
eg. a path vs a (beautiful and long) path
Ideally this could capture possible compression operations automatically
A learnable function to score the probablistic path
good: f(beautiful and long; a b and l path; len(b and l) )
bad: f(a good thing and a bad thing)
we measure these against: faithfulness? ROUGE, grammar
After decoding with the algorithm, any minimal path could come with a few options.
These options can be nested. 
With beam search, we should be able to get a few **different** path and each of them with many options.
Macro-BS will mostly focus on these path; micro-BS will focus on small option chunks.
a good man with a good mind => a (good) man (with a (good) mind)
a (55-year-old) man (, 55,) went to ...   competing/colliding
