
import os
from collections import defaultdict
import pickle
from operator import le
from tqdm import tqdm
import textstat

from collections import UserDict
from pyvis.network import Network
import networkx as nx

from src.recom_search.model.beam_state import BeamState
from src.recom_search.model.util import pnum, tokenizer
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
        if '</s>' in pointer.token_str or '.' in pointer.token_str:
            net.add_node(
                pointer.uid, label=f"{pointer.token_str}", group=group_num, color='red')
        else:
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
        try:
            net.add_edge(span_b.prefix_node.uid, f"merge_{MERGE_CNT}")
            net.add_edge(f"merge_{MERGE_CNT}", span_b.suffix_node.uid)
        except AssertionError:
            pass
        MERGE_CNT += 1
        seen_merge.add(uid_span_b)


def draw_nodes(net, nodes, group_num):
    for node in nodes:
        form = "{:.1f}".format(node['score']) 
        net.add_node(
            node['uid'], label=f"{node['text']}\n{ form  }", shape='dot',group=group_num, size=8)


def draw_edges(net, edges, group_num):
    for edge in edges:
        if edge['seen']:
            net.add_edge(edge['src'], edge['tgt'],  arrowStrikethrough=False)
        else:
            net.add_edge(edge['src'], edge['tgt'], arrowStrikethrough=False)


def viz_result(generated_outputs: List[BeamState]):
    for go in generated_outputs:
        print(go)
    net = Network(height='1500px', width='100%', directed=True)
    # net.repulsion(central_gravity=0.2, spring_length=30)

    # net.toggle_stabilization(False)
    # first set all nodes and edges
    for idx, go in enumerate(generated_outputs):
        nodes, edges = go.visualization()
        draw_nodes(net,nodes, idx)
        draw_edges(net,edges,idx)
    return net


if __name__ == "__main__":
    # execute only if run as a script
    files = os.listdir('vizs')
    files = [f for f in files if f.endswith('.pkl') and f.startswith('best')]
    for f in tqdm(files):
        name = f.split('.')[0]
        with open(f"vizs/{f}", 'rb') as fd:
            finished = pickle.load(fd)
        net = viz_result(finished)
        net.show(f"vizs/html/{name}.html")
