import faiss
import sys
import numpy as np
import numpy.linalg as la
import torch.nn as nn
import matplotlib.pyplot as plt
from numpy import log
from pyitlib import discrete_random_variable as drv
from bisect import bisect
from scipy.special import digamma
from sklearn.neighbors import BallTree, KDTree
from mighty.monitor.mutual_info.npeet import *
from mighty.monitor.mutual_info.kmeans import *
from mine.models.mine import Mine
from mine.models.mine import MutualInformationEstimator

#----
#ESTIMATORS
#----

#from entropy_estimators.py // called mi()
def kraskov(x, y, z=None, k=3, base=2, alpha=0):
    assert len(x) == len(y),
    assert k <= len(x) - 1,
    
    x, y, = np.asarray(x), np.asarray(y)
    x, y = x.reshape(x.shape[0], -1), y.reshape[0], -1)
    x = add_noise(x)
    y = add_noise(y)
    points = [x, y]
    if z is not None:
        z = np.asarray(z)
        z = z.reshape(z.shape[0], -1)
        points.append(z)
    points = np.hstack(points)
    
    #find newrest neighbors in joint space, p=inf means max-norm
    tree = build_tree(points)
    dvec = query_neighbors(tree, points, k)
    if z is None:
        a, b, c, d = avgdigamma(x, dvec), avgdigamma(y, dvec), digamma(k), digamma(len(x))
        if alpha > 0:
            d += lnc_correction(tree, points, k, alpha)
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = avgdigamma(xz, dvec), avgdigamma(yz, dvec), avgdigamma(z, dvec), digamma(k)    
    return (-a - b + c + d) / log(base)

#from Linear_Function.py
def kmeans(x, y) -> float:
    #create quantized data for kmeans
    xKmeans = _quantize(x)
    ykMeans = _quantize(y)
    return MutualInfo.to_bits(mutual_info_score(xKmeans, yKmeans))

def mine(x, y, fsize=100):
    dimX, dimY = len(x), len(y)
    statistics_network = nn.Sequential(nn.Linear(dimX+dimY, fsize), nn.ReLU(), nn.Linear(fsize, fsize), nn.ReLU(), nn.Linear(fsize, fsize), nn.ReLU(), nn.Linear(fsize, 1))
    mine = Mine(T = statistics_network, loss = 'fdiv', method = 'concat')
    return mine.optimize(x, y, batch_size=100, iters=100)

#----
#CORRECTION FUNCTIONS
#----

#from entropy_estimators.py
def lnc_correction(tree, points, k, alpha):
    e = 0
    n_sample = points.shape[0]
    for point in points:
        #Find k-nearest neighbors in joint space, p=inf means max norm
        knn = tree.query(point[None, :], k=k+1, return_distance=False)[0]
        knn_points = points[knn]
        #Subtract mean of k-nearest neighbor points
        knn_points = knn_points - knn_points[0]
        #Calculate covariance matrix of k-nearest neighbor points, obtain eigenvectors
        covr = knn_points.T @ knn_points / k
        _, v = la.eig(covr)
        #Calculate PCA-bounding box using eigenvectors
        V_rect = np.log(np.abs(knn_points @ v).max(axis=0)).sum()
        #Calculate the volume of original box
        log_knn_dist = np.log(np.abs(knn_points).max(axis=0)).sum()
        #Perform local non-uniformity checking and update correction term
        if V_rect < log_knn_dist + np.log(alpha):
            e += (log_knn_dist - V_rect) / n_sample
    return e

#----
#UTILITY FUNCTIONS
#----

#from entropy_estimators.py
def add_noise(x, intens=1e-10):
    #small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)

#from entropy_estimators.py
def query_neighbors(tree, x, k):
    return tree.query(x, k=k+1)[0][:, k]

#from entropy_estimators.py
def count_neighbors(tree, x, r):
    return tree.query_radius(x, r, count_only=True)

#from entropy_estimators.py
def avgdigamma(points, dvec):
    #This part finds the number of neighbors in some radius in the marginal space
    #returns expectation value of <psi(nx)>
    tree = build_tree(points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))

#from entropy_estimators.py
def build_tree(points):
    if points.shape[1] >= 20:
        return BallTree(points, metric='chebyshev')
    return KDTree(points, metric='chebyshev')

#from Linear_Function.py
def _quantize(activations: torch.FloatTensor) -> np.ndarray:
    n_bins = 10
    model = cluster.MiniBatchKMeans(n_clusters=n_bins, batch_size=100)
    labels = model.fit_predict(activations)
    return labels
