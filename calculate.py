import numpy as np
import torch.nn as nn
from scipy.special import digamma
from sklearn.metrics import mutual_info_score
import utils

#----
#ESTIMATORS
#----

#from entropy_estimators.py // called mi()
def kraskov(x, y, z=None, k=3, base=2, alpha=0):
    """Mutual information on x and y (conditioned on z if z is not None)
       x, y should be lists of vectors, i.e. x = [[1.3], [3.7], [5.1], [2.4]]
       if x is a one-dimensional scalar and we have four samples
    """
    assert len(x) == len(y) #Arrays should have same length
    assert k <= len(x) - 1  #Set k smaller than num. samples - 1
    
    x, y, = np.asarray(x), np.asarray(y)
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    x = utils.add_noise(x)
    y = utils.add_noise(y)
    points = [x, y]
    if z is not None:
        z = np.asarray(z)
        z = z.reshape(z.shape[0], -1)
        points.append(z)
    points = np.hstack(points)
    
    #find newrest neighbors in joint space, p=inf means max-norm
    tree = utils.build_tree(points)
    dvec = utils.query_neighbors(tree, points, k)
    if z is None:
        a, b, c, d = utils.avgdigamma(x, dvec), utils.avgdigamma(y, dvec), utils.digamma(k), digamma(len(x))
        if alpha > 0:
            d += utils.lnc_correction(tree, points, k, alpha)
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = utils.avgdigamma(xz, dvec), utils.avgdigamma(yz, dvec), utils.avgdigamma(z, dvec), digamma(k)    
    return (-a - b + c + d) / np.log(base)

#from Linear_Function.py
def kmeans(x, y) -> float:
    #create quantized data for kmeans
    xKmeans = utils._quantize(x)
    yKmeans = utils._quantize(y)
    return utils.to_bits(mutual_info_score(xKmeans, yKmeans)) #pytorch-mighty/monitor/mutual_info.py 

def mine(x, y, fsize=100):
    dimX, dimY = len(x), len(y)
    statistics_network = nn.Sequential(nn.Linear(dimX+dimY, fsize), nn.ReLU(), nn.Linear(fsize, fsize), nn.ReLU(), nn.Linear(fsize, fsize), nn.ReLU(), nn.Linear(fsize, 1))
    mine = utils.Mine(T = statistics_network, loss = 'fdiv', method = 'concat')
    return mine.optimize(x, y, batch_size=100, iters=100)



