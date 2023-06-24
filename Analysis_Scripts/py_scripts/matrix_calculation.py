import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from scipy.special import kl_div
from collections import Counter
import os

def get_pairs(trace):
    trace = trace.split()
    return [(trace[i], trace[i+1]) for i in range(len(trace) - 1)]

def construct_adjacency_matrix(trace1, trace2, epsilon=1e-10):
    union = list(set(trace1.split()).union(set(trace2.split())))
    t1_t2_matrix = np.zeros((len(union), len(union)))
    t2_t1_matrix = np.zeros((len(union), len(union)))

    for i in range(len(union)):
        for j in range(len(union)):
            if i != j:
                t1_t2_matrix[i][j] = 1
                t2_t1_matrix[i][j] = 1

    pairs1 = get_pairs(trace1)
    pairs2 = get_pairs(trace2)
    pair_freq1 = Counter([tuple(pair) for pair in pairs1])
    pair_freq2 = Counter([tuple(pair) for pair in pairs2])

    for pair, freq in pair_freq1.items():
        t1_t2_matrix[union.index(pair[0])][union.index(pair[1])] += freq
    for pair, freq in pair_freq2.items():
        t2_t1_matrix[union.index(pair[0])][union.index(pair[1])] += freq

    t1_t2_matrix += epsilon
    t2_t1_matrix += epsilon
    t1_t2_matrix = t1_t2_matrix / t1_t2_matrix.sum(axis=1, keepdims=True)
    t2_t1_matrix = t2_t1_matrix / t2_t1_matrix.sum(axis=1, keepdims=True)

    return t1_t2_matrix, t2_t1_matrix

def compute_kl_divergence(trace1, trace2):
    t1_t2_kld = np.sum(kl_div(trace1.flatten(), trace2.flatten()))
    t2_t1_kld = np.sum(kl_div(trace2.flatten(), trace1.flatten()))
    return t1_t2_kld, t2_t1_kld

def compute_js_divergence(trace1, trace2):
    js_divergence = jensenshannon(trace1.flatten(), trace2.flatten())
    return js_divergence

def run_all_traces(all_traces):
    num_traces = len(all_traces)
    print(f'Number of traces: {num_traces}')

    kld_matrix = np.zeros((num_traces, num_traces))
    jsd_matrix = np.zeros((num_traces, num_traces))

    for i in range(num_traces):
        print(f'Computing kld and jsd for trace {i}')
        for j in range(i+1, num_traces):
            matrix1, matrix2 = construct_adjacency_matrix(all_traces[i], all_traces[j])
            kld_12, kld_21 = compute_kl_divergence(matrix1, matrix2)
            jsd = compute_js_divergence(matrix1, matrix2)
            kld_matrix[i][j] = kld_12
            kld_matrix[j][i] = kld_21
            jsd_matrix[i][j] = jsd
            jsd_matrix[j][i] = jsd

    return kld_matrix, jsd_matrix

def main():
    # read data
    with open("all_analysis_data.txt", "r") as f:
        all_traces = f.read().split('\n')[:-1]

    with open("malware_api_class_master_labels.csv", "r") as g:
        all_labels = g.read().split('\n')[:-1]

    # run calculations
    kld_matrix, jsd_matrix = run_all_traces(all_traces)

    # write out to csv file 
    np.savetxt("kld_matrix.csv", kld_matrix, delimiter=",")
    np.savetxt("jsd_matrix.csv", jsd_matrix, delimiter=",")

if __name__ == "__main__":
    main()
