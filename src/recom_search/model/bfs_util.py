from collections import defaultdict
from typing import List
import random
random.seed(2021)
class NewHash():
    def __init__(self, ngram: int = 5) -> None:
        self.data = defaultdict(list)   # key: a_b_c: val: [uid_a, uid_b, uid_c]
        self.ngram = ngram
        self.uid_map = {}   # if the value is a string, do while; else it is the node
    
    def set_node(self, uid, node):
        self.uid_map[uid] = node

    def const_key(self, token_ids:List[int]):
        tokens = token_ids[-self.ngram:]
        token_str = [str(x) for x in tokens]
        k = "_".join(token_str)
        return k
    
    def retrieve_node(self, q_uid):
        
        key = q_uid
        while isinstance(key,str):
            key = self.uid_map[key]
        return key  # key is actually a node

    def find_root_node_uid(self, q_uid ):
        key = q_uid
        while True:
            tmp_key = self.uid_map[key]
            if isinstance(tmp_key, str):
                key = tmp_key
                continue
            else:
                return key

    def retrieve_group_nodes(self, keys):
        # keys contain some UIDs
        # return updated keys and retrieved nodes
        candidates = [self.find_root_node_uid(k) for k in keys ]
        candidates = [i for n, i in enumerate(candidates) if i not in candidates[:n]]
        cand_nodes = [ self.uid_map[x] for x in candidates]
        return candidates, cand_nodes

    def retrieve_ngram_nodes(self, key_ngram):
        if key_ngram in self.data:
            candidates = self.data[key_ngram]
            updated_cands, nodes = self.retrieve_group_nodes(candidates)
            self.data[key_ngram] = updated_cands
            return nodes
        else:
            return []

    def query(self, token_ids: List[int]):
        # get the last n tokens
        if len(token_ids) < self.ngram:
            return []
        k = self.const_key(token_ids)
        return self.retrieve_ngram_nodes(k)
    
    def replace_node(self, master_node_uid, del_node_uid):
        print(f"rep: {del_node_uid} {master_node_uid}")
        self.uid_map[del_node_uid] = master_node_uid

    def add_helper(self, par_node, new_node):
        """
        Add all the combination of par_node + new_node token to the hash
        """
        # par_node : the parent node

        def dfs(node, depth):
            if not node:
                return []
            if depth == self.ngram:
                return [[node.token_idx]]
            prevs = node.prev
            _updated_cand, prevs = self.retrieve_group_nodes(prevs)
            node.prev = _updated_cand
            outputs = []
            for p in prevs:
                many = dfs(p, depth+1)
                for one in many:
                    outputs.append(one + [node.token_idx])
            return outputs
        all_probable_paths = dfs(par_node, 2)
        all_probable_paths = [x + [new_node.token_idx]
                              for x in all_probable_paths if len(x) == self.ngram-1]

        cnt = 0
        for p in all_probable_paths:
            key = self.const_key(p)
            self.data[key].append(new_node.uid)
            cnt += 1

