d = '/mnt/data1/jcxu/lattice-sum/output/data/sum_xsum_bs_16_35_False_0.4_False_False_4_5_zip_-1_0.0_0.9'
d = '/mnt/data1/jcxu/lattice-sum/output/data/sum_xsum_dbs_16_35_False_0.4_False_False_4_5_zip_-1_0.0_0.9'
dirs = [
    '/mnt/data1/jcxu/lattice-sum/output/data/mtn1_zh-en_bs_8_-1_False_0.4_False_False_4_5_zip_-1_0.0_0.9',
'/mnt/data1/jcxu/lattice-sum/output/data/mtn1_zh-en_dbs_8_-1_False_0.4_False_False_4_5_zip_-1_0.0_0.9', 
'/mnt/data1/jcxu/lattice-sum/output/data/mtn1_fr-en_bs_8_-1_False_0.4_False_False_4_5_zip_-1_0.0_0.9',
'/mnt/data1/jcxu/lattice-sum/output/data/mtn1_fr-en_dbs_8_-1_False_0.4_False_False_4_5_zip_-1_0.0_0.9',
'/mnt/data1/jcxu/lattice-sum/output/data/mt1n_en-fr_bs_8_-1_False_0.4_False_False_4_5_zip_-1_0.0_0.9',
'/mnt/data1/jcxu/lattice-sum/output/data/mt1n_en-fr_dbs_8_-1_False_0.4_False_False_4_5_zip_-1_0.0_0.9'

]
import os
import pickle

from src.recom_search.evaluation.analysis import derive_path, viz_result

files = os.listdir(d)

def study_bs(generated_outputs):
    
    if len(generated_outputs) == 0:
        return {}

    # first set all nodes and edges
    all_nodes = {}
    all_edges = {}

    for idx, go in enumerate(generated_outputs):
        nodes, edges = go.visualization()
        all_nodes.update(nodes)
        all_edges.update(edges)

    all_paths, all_eos, all_degree_mat, sos_key = derive_path(
        all_nodes, all_edges, True)
    truncate_len = min([len(x.tokens) for x in all_paths])
    # edges" srcid tgtid
    edge_info = list(all_edges.keys())
    def limit_dfs(node_id, depth):
        if depth == truncate_len:
            return 1
        cnt = 1
        targets = [ x.split('_')[1] for x in edge_info if x.startswith(node_id)]
        for t in targets:
            cnt += limit_dfs(t, depth+1)
        return cnt
    total = limit_dfs(sos_key, 1)
    # print(total)
    # print(truncate_len * 16)
    return total / (truncate_len * 8)
for d in dirs:
    bag = []
    for f in files:
        with open(os.path.join(d,f),'rb') as fd:
            data = pickle.load(fd)
        rate = study_bs(data.ends)
        bag.append(rate)
    # print(bag)
    import statistics
    print(statistics.mean(bag))
    