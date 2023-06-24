import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import os

def create_graph(matrix, percentile):
    num_nodes = matrix.shape[0]
    k = max(1, int(np.ceil(num_nodes * percentile / 100))) 

    nbrs = NearestNeighbors(n_neighbors=k, metric='precomputed').fit(matrix)
    distances, indices = nbrs.kneighbors(matrix)

    G = nx.Graph()

    for i in range(num_nodes):
        for j in range(k):
            G.add_edge(str(i), str(indices[i][j]), weight=distances[i][j])
            
    return G

def main():
    # read in csv files
    kld_matrix = np.genfromtxt("kld_matrix.csv", delimiter=",")
    jsd_matrix = np.genfromtxt("jsd_matrix.csv", delimiter=",")

    # create graphs
    G_jsd = nx.from_numpy_matrix(jsd_matrix)
    G_kld = nx.from_numpy_matrix(kld_matrix)

    # compute minimum spanning trees
    mst_jsd = nx.minimum_spanning_tree(G_jsd)
    mst_kld = nx.minimum_spanning_tree(G_kld)

    # write minimum spanning trees to gml files
    nx.write_gml(mst_jsd, "jsd_mst.gml")
    nx.write_gml(mst_kld, "kld_mst.gml")

    # create graphs for 1%, 2% and 5%
    for i in [1, 1.5, 2, 2.5, 3]:
        G_jsd = create_graph(jsd_matrix, i)
        G_kld = create_graph(kld_matrix, i)
        nx.write_gml(G_jsd, f"jsd_{i}.gml")
        nx.write_gml(G_kld, f"kld_{i}.gml")

if __name__ == "__main__":
    main()
