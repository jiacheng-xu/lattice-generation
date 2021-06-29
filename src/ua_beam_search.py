"""
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
"""
class SuperBeam(object):
    """
    Beam data structure. Maintains a list of scored elements like a Counter, but only keeps the top n
    elements after every insertion operation. Insertion is O(n) (list is maintained in
    sorted order), access is O(1). Still fast enough for practical purposes for small beams.
    """
    def __init__(self, size):
        self.size = size
        self.elts = []
        self.scores = []
        self.backup = []

    def __repr__(self):
        return "Beam(" + repr(list(self.get_elts_and_scores())) + ")"
    
    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.elts)

def test_beam():
    print("test")
    # for each step 
    beam = SuperBeam(5)


if __name__ == '__main__':
    test_beam()
