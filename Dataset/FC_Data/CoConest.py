import os
import numpy as np
import scipy.io
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt

def err_comp_prune(link_mat, data, tree_error):
    errors = np.zeros(link_mat.shape[0])
    
    for i in range(link_mat.shape[0]):
        cluster_indices = fcluster(link_mat, i + 1, criterion='maxclust')
        error = tree_error(data, cluster_indices)
        errors[i] = error
    
    min_error_index = np.argmin(errors)
    pruned_link_mat = link_mat[:min_error_index + 1, :]
    
    prune_struct = {'link': pruned_link_mat, 'errors': errors}
    return prune_struct

def example_tree_error(data, cluster_indices):
    unique_clusters = np.unique(cluster_indices)
    error = 0
    for cluster in unique_clusters:
        cluster_data = data[cluster_indices == cluster]
        cluster_error = np.sum((cluster_data - cluster_data.mean(axis=0)) ** 2)
        error += cluster_error
    return error

def tree2IDX(prune_struct, k):
    link_mat = prune_struct['link']
    idx = fcluster(link_mat, k, criterion='maxclust')
    parc = np.zeros((k, k))
    
    for i in range(1, k + 1):
        for j in range(1, k + 1):
            if i == j:
                parc[i - 1, j - 1] = np.sum(idx == i)
            else:
                parc[i - 1, j - 1] = np.sum((idx == i) & (idx == j))
    
    return idx, parc

def plot_dendrogram(link_mat, title, filename):
    plt.figure(figsize=(10, 7))
    dendrogram(link_mat)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def create_prune_tree(parc_name, data_path, out_fold, link_method='average', dist_metric='euclidean', tree_error=example_tree_error, kseq=[5, 10, 15, 20]):
    print(f"parc_name: {parc_name}")
    print(f"data_path: {data_path}")
    print(f"out_fold: {out_fold}")
    print(f"link_method: {link_method}")
    print(f"dist_metric: {dist_metric}")

    data = scipy.io.loadmat(data_path)['average_fc_avg_ico4']

    link_mat = linkage(data, method=link_method, metric=dist_metric)

    prune_struct = err_comp_prune(link_mat, data, tree_error)
    prune_struct['link'] = link_mat

    tree_results_dir = os.path.join(out_fold, "TreeResults")
    parcellations_dir = os.path.join(tree_results_dir, "parcellations")
    os.makedirs(tree_results_dir, exist_ok=True)
    os.makedirs(parcellations_dir, exist_ok=True)

    prune_struct_path = os.path.join(tree_results_dir, f"{parc_name}_prune_struct.mat")
    scipy.io.savemat(prune_struct_path, {'prune_struct': prune_struct})

    plot_dendrogram(link_mat, f"{parc_name} Dendrogram", os.path.join(tree_results_dir, f"{parc_name}_dendrogram.png"))

    for k in kseq:
        idx, parc = tree2IDX(prune_struct, k)
        parc_path = os.path.join(parcellations_dir, f"{parc_name}_{k}_parc.mat")
        scipy.io.savemat(parc_path, {'parc': parc})

    return 1

parc_name = 'CoCoNest'
data_path = '/work/users/y/i/yifzhang/zhengwu/EPIC_Data/fitbit_Data/average_fc_avg_ico4.mat'
out_fold = '/work/users/y/i/yifzhang/zhengwu/EPIC_Data/fitbit_Data/output'
kseq = [5, 10, 15, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 450, 475, 500]


create_prune_tree(parc_name, data_path, out_fold, kseq=kseq)
