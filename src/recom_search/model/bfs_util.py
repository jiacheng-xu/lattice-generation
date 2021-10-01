from collections import defaultdict
from typing import List

from src.recom_search.model.beam_state import BeamNode

class HashedGen():
    def __init__(self, ngram: int = 5) -> None:
        self.data = defaultdict(list)
        self.ngram = ngram

    def const_key(self, token_ids):
        tokens = token_ids[-self.ngram:]
        token_str = [str(x) for x in tokens]
        k = "_".join(token_str)
        return k

    def query(self, token_ids: List[int]):
        # get the last n tokens
        if len(token_ids) < self.ngram:
            return []
        k = self.const_key(token_ids)
        if k in self.data:
            return self.data[k]
        else:
            return []

    def add(self, node):
        tokens = node.all_token_idx
        if len(tokens) < self.ngram:
            return
        k = self.const_key(tokens)
        self.data[k].append(node)

    def add_helper(self, par_node, new_node):
        # par_node : the parent node

        def dfs(node: BeamNode, depth):
            if not node:
                return []
            if depth == self.ngram:
                return [[node.token_idx]]
            prevs = node.prev
            outputs = []
            for p in prevs:
                many = dfs(p, depth+1)
                for one in many:
                    outputs.append(one + [node.token_idx])
            return outputs
        all_probable_paths = dfs(par_node, 1)
        all_probable_paths = [x + [new_node.token_idx]
                              for x in all_probable_paths if len(x) == self.ngram]

        cnt = 0
        for p in all_probable_paths:
            key = self.const_key(p)
            self.data[key].append(new_node)
            cnt += 1
        # logging.debug(f"{cnt} added to Hash.")

