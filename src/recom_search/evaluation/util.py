
from collections import defaultdict
import pickle
from operator import le
from tqdm import tqdm
import textstat

from collections import UserDict
from pyvis.network import Network
import networkx as nx

from src.recom_search.model.beam_state import BeamState
from src.recom_search.model.util import tokenizer
from typing import List

MERGE_CNT = 0


def draw_one_summary(net, final_beam, group_num, seen_merge):
    global MERGE_CNT
    pointer = final_beam
    last = None
    all_merges = []
    T = len(pointer.token_full)
    # cumulative ends
    nodes, edges = {}, {}
    degree = defaultdict(int)
    while pointer:
        nodes[pointer.uid] = {
            'text': pointer.token_str,
            'special': False,  # EOS SOS Merge
            'group': group_num,
            't': T
        }
        T -= 1
        if last:
            edges[f"{pointer.uid}_{last}"] = 1

        net.add_node(
            pointer.uid, label=f"{pointer.token_str}", group=group_num, shape='text')

        if last:
            net.add_edge(pointer.uid, last, weight=0.1,
                         arrowStrikethrough=False)

        if pointer.merge != []:
            all_merges += pointer.merge

        last = pointer.uid
        pointer = pointer.prev

    for merge_pair in all_merges:
        span_a, span_b = merge_pair
        uid_span_b = '_'.join([str(x) for x in span_b.tokens])
        if uid_span_b in seen_merge:
            continue
        # span_a is already in use?
        net.add_node(f"merge_{MERGE_CNT}", label=tokenizer.decode(
            span_b.tokens), group=group_num, color='black')
        net.add_edge(span_b.prefix_node.uid, f"merge_{MERGE_CNT}")
        net.add_edge(f"merge_{MERGE_CNT}", span_b.suffix_node.uid)
        MERGE_CNT += 1
        seen_merge.add(uid_span_b)


def viz_result(generated_outputs: List[BeamState]):
    for go in generated_outputs:
        print(go)
    net = Network(height='1500px', width='100%', directed=False)
    net.repulsion(central_gravity=0.1, spring_length=30)
    seen_merge = set()
    # net.toggle_stabilization(False)
    # first set all nodes and edges
    for idx, go in enumerate(generated_outputs):
        draw_one_summary(net, go, idx, seen_merge)

    net.show('vizs/nx.html')


if __name__ == "__main__":
    # execute only if run as a script
    with open('vizs/00.pkl', 'rb') as fd:
        finished = pickle.load(fd)
    viz_result(finished)
