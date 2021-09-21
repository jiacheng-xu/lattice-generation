
import os
from collections import defaultdict
import pickle

from tqdm import tqdm
import textstat

from collections import UserDict
from pyvis.network import Network
import networkx as nx

from src.recom_search.model.beam_state import BeamNode
from src.recom_search.model.util import pnum, tokenizer
from typing import List


def draw_nodes(net, nodes, group_num):
    for node in nodes:
        # form = "{:.1f}".format(node['score']) 
        net.add_node(
            node['uid'], label=f"{node['text']}", shape='dot',group=group_num, size=8)


def draw_edges(net, edges, group_num):
    for edge in edges:
        form = "{:.1f}".format(edge['score']) 
        net.add_edge(edge['src'], edge['tgt'], title=form,  arrowStrikethrough=False)


def viz_result(generated_outputs: List[BeamNode]):
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
