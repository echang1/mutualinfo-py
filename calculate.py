import numpy as np
import math
import torch
import torch.nn as nn
from scipy.special import digamma
from sklearn.metrics import mutual_info_score
import utils

#from entropy_estimators.py
def kraskov(x, y, z=None, k=3, base=2, alpha=0):
    """Mutual information on x and y (conditioned on z if z is not None)
       x, y should be lists of vectors, i.e. x = [[1.3], [3.7], [5.1], [2.4]]
       if x is a one-dimensional scalar and we have four samples
    """
    assert len(x) == len(y) #Arrays should have same length
    assert k <= len(x) - 1  #Set k smaller than num. samples - 1
    
    x, y = np.asarray(x), np.asarray(y)
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
        a, b, c, d = utils.avgdigamma(x, dvec), utils.avgdigamma(y, dvec), digamma(k), digamma(len(x))
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

#fro Linear_Function.py
def mine(x, y, fsize=100, batch_size=100):
    dimX, dimY = len(x[0]), len(y[0])
    statistics_network = nn.Sequential(nn.Linear(dimX+dimY, fsize), nn.ReLU(), nn.Linear(fsize, fsize), nn.ReLU(), nn.Linear(fsize, fsize), nn.ReLU(), nn.Linear(fsize, 1))
    mine = utils.Mine(T=statistics_network, loss = 'fdiv', method = 'concat')
    return mine.optimize(x, y, batch_size=batch_size, iters=100)

#from mine.models.mine.py
class Mine(nn.Module):
    def __init__(self, T, loss='mine', alpha=0.01, method=None):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.method = method

        if method == 'concat':
            if isinstance(T, nn.Sequential):
                self.T = utils.CustomSequential(utils.ConcatLayer(), *T)
            else:
                self.T = utils.CustomSequential(utils.ConcatLayer(), T)
        else:
            self.T = T

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        t = self.T(x, z).mean()
        t_marg = self.T(x, z_marg)

        if self.loss in ['mine']:
            second_term, self.running_mean = utils.ema_loss(t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in['mine_biased']:
            second_term = torch.logsumexp(t_marg, 0) - math.log(t_marg.shape[0])

        return -t + second_term

    def mi(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi

    def optimize(self, X, Y, iters, batch_size, opt=None):

        if opt is None:
            opt = torch.optim.Adam(self.parameters(), lr=1e-4)
           
        for iter in range(1, iters + 1):
            mu_mi = 0
            for x, y in utils.batch(X, Y, batch_size):
                opt.zero_grad()
                loss = self.forward(x, y)
                loss.backward()
                opt.step()
                mu_mi -= loss.item()
            if iter % (iters // 3) == 0:
                pass
        final_mi = self.mi(X, Y)
        return final_mi.item()



