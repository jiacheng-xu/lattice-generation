
from tqdm import tqdm
import textstat

from collections import UserDict
from pyvis.network import Network
import networkx as nx

def viz_result():
    net = Network(height='1500px', width='100%', )
    batch = 5
    T = 10
    node_id = 0
    for t in range(T):

        for j in range(batch):
            net.add_node(f"{t}-{j}") 

            if t >0:
                for k in range(batch):
                    net.add_edge(f"{t-1}-{k}", f"{t}-{j}")

    net.show('nx.html')

viz_result()