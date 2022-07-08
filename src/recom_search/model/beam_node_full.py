from src.recom_search.model.bfs_util import HashObject
from src.recom_search.model.beam_node import BeamNode
from typing import List

class BeamNodeFull(BeamNode):
    def __init__(self, hash:HashObject, prob: float, token_idx: int, prev: List, prev_score: List, min_len: int, finished: bool = False, master_node_uid= None) -> None:
        super().__init__(prob, token_idx, prev, prev_score, min_len, finished)
        self.hash = hash
        self.master_node_uid = master_node_uid or self.uid
        self.set_full()
        assert self.all_token_idx
        assert self.all_score
        assert self.length

        self.has_finished()
        self.hash.set_node(self.uid, self)

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

