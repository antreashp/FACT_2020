import matplotlib
matplotlib.rc("xtick", labelsize = 24)
matplotlib.rc("ytick", labelsize = 24)
matplotlib.rc("axes", titlesize = 48)
matplotlib.rc("axes", labelsize = 48)
matplotlib.rc("lines", markersize = 16)

import pandas as pd 
import numpy as np
from sklearn.mixture import GaussianMixture 
import matplotlib.pyplot as plt

def cluster(x, data_rep, num_clusters):
    model = GaussianMixture(n_components=num_clusters, random_state=0)
    labels = model.fit_predict(data_rep)

    n = x.shape[0]
    cluster = -1.0 * np.ones((n))

    indices = [[]] * num_clusters
    centers = [[]] * num_clusters
    means = [[]] * num_clusters

    for i in range(num_clusters):
        indices[i] = []
        for j in range(n):
            if labels[j] == i:
                cluster[j] = i
                indices[i].append(j)
        means[i] = np.mean(x[indices[i], :], axis = 0)
        centers[i] = np.mean(data_rep[indices[i], :], axis = 0)
        
    centers = np.array(centers)
    means = np.array(means)

    return means, centers, indices, cluster  
    

def plot_groups(data_rep, num_clusters, centers, cluster, contour = None, name = "plot_groups.png"):

    fig, ax = plt.subplots(figsize=(20, 10))    
    plt.scatter(data_rep[:, 0], data_rep[:, 1], c = cluster, cmap = plt.cm.coolwarm)

    for i in range(num_clusters):
        plt.text(centers[i, 0], centers[i, 1], str(i), fontsize = 72)
        
    if contour is not None:
        feature_0 = contour[0]
        feature_1 = contour[1]
        map = contour[2]
        plt.contour(feature_0, feature_1, map)
        plt.colorbar()

    plt.savefig(name)
    plt.show()
    plt.close()