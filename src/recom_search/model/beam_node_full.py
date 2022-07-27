from src.recom_search.model.bfs_util import HashObject
from src.recom_search.model.beam_node import BeamNode
from typing import List

class BeamNodeFull(BeamNode):
    def __init__(self, hash:HashObject, prob: float, token_idx: int, prev: List, prev_score: List, min_len: int=10, finished: bool = False, master_node_uid= None) -> None:
        super().__init__(prob, token_idx, prev, prev_score, min_len, finished)
        self.hash = hash
        self.master_node_uid = master_node_uid or self.uid
        self.set_full()
        assert self.all_token_idx
        assert self.all_score
        assert self.length

        self.has_finished()
        self.hash.set_node(self.uid, self)

    def set_full(self):
        """
        Everytime a substructure is modified, we need to update the tokens and scores
        """
        tokens = [self.token_idx]
        scores = [self.score]
        prev = self.prev
        # print(self.prev)
        while prev:
            prev = prev[0]
            prev = self.hash.retrieve_node(prev)
            tokens.append(prev.token_idx)
            scores.append(prev.score)
            prev = prev.prev
        self.all_score = scores[::-1]
        self.all_token_idx = tokens[::-1]
        self.length = len(tokens)

    def get_antecedent(self, step=10):
        """
        Get all antecedents of the node. 
        """
        antecedents = []

        prev = self.prev  # prev is a list
        _, prev = self.hash.retrieve_group_nodes(prev)
        cnt = 0
        while prev:
            antecedents += prev
            new_prev = []
            for p in prev:
                new_prev += p.prev
            _, new_prev = self.hash.retrieve_group_nodes(new_prev)
            # new_prev = list(set(new_prev))
            new_prev = [x for x in new_prev if x not in antecedents]
            prev = new_prev
            cnt += 1
            if step is not None and cnt >= step:
                break
        return antecedents

    def add_prev_node(self, node_id, score):
        """
        self: a b c d  a   f g
        node: a b c d  x y 

        """
        # loop detection
        # check if self is the ancedant node of "node"
        if self in self.hash.retrieve_node(node_id).get_antecedent() or self.hash.find_root_node_uid(self.uid)  == self.hash.find_root_node_uid(node_id) or (node_id in self.prev):
            # do not add prev node
            return

        self.prev.append(node_id)
        self.prev_score.append(score)


    def visualization(self):
        nodes, edges = {}, {}
        seen = {}

        def dfs(node: BeamNodeFull):
            if not node:
                return
            node_uid = self.hash.find_root_node_uid(node.uid)
            if node_uid in seen:
                return
            seen[node_uid] = True
            node = self.hash.retrieve_node(node_uid)
            my_prev, my_prev_score = node.prev, node.prev_score
            for p, ps in zip(my_prev, my_prev_score):
                p_uid = self.hash.find_root_node_uid(p)
                # p_node = self.hash.retrieve_node(p_uid)
                edge_info = {
                    'src': p_uid,
                    'tgt': node_uid,
                    'score': ps
                }
                edges[f"{p_uid}_{node_uid}"] = edge_info
                # edges.append(edge_info)

            nodes[node.uid] = {
                'uid': node_uid,
                'text': node.token_str,
                'tok_idx': node.token_idx
            }

            prevs = node.prev
            _, prevs = self.hash.retrieve_group_nodes(prevs)
            for p in prevs:
                dfs(p)
        dfs(self)
        return nodes, edges
    
    def get_tokens_match_suffix(self, inp_suffix_tokens: List[int]):
        """
        suffix_tokens is the target suffix to match
        greedily pick the prev[0] for rest of sequence
        possible improvement, return all possible and do ANY for heuristic math
        """
        reversed_tokens = []
        suffix_tokens = inp_suffix_tokens.copy()
        
        prev = [[self.uid]]
        while prev and suffix_tokens:
            last_target_token_idx = suffix_tokens.pop(-1)
            # print('-----')
            # print(tokenizer.decode(last_target_token_idx))
            new_prev = []
            for list_p in prev:
                # p is a list of UIDs, the last one is the latest one
                p = list_p[-1]
                node = self.hash.retrieve_node(p)
                # print(node.token_str)
                token = node.token_idx
                if token == last_target_token_idx:
                    # _, tmp_prev = self.hash.retrieve_group_nodes(node.prev)
                    new_prev += [list_p + [x] for x in node.prev] 

            reversed_tokens.append(last_target_token_idx)
            prev = new_prev
            if not prev:
                raise ValueError("Not found!")
        matched_suffix_node_ids = prev[0][:-1]
        prev = prev[0]
        while prev:
            node_uid = prev[0]
            node = self.hash.retrieve_node(node_uid)
            reversed_tokens.append(node.token_idx)
            prev = node.prev

        return reversed_tokens[::-1], matched_suffix_node_ids


    def get_tokens_str(self):
        out = [self.token_str]
        prev = self.prev
        while prev:
            prev = prev[0]
            prev = self.hash.retrieve_node(prev)
            out.append(prev.token_str)
            prev = prev.prev
        out = out[::-1]
        return '-'.join(out)