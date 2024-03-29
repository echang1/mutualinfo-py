import numpy as np
import torch

#from mutual_infomation_data.py
def generate_linear_data(stdN, n_bins=10, N=500, stdX=10, dimX=2, dimY=3):
    W = np.random.rand(dimY,dimX)
    for i in range(dimY):
        for j in range(dimX):
            W[i][j] = i+j
    
    X = np.zeros((dimX,N))
    noise = np.zeros((dimY,N))
    for i in range(N):
        X[:,i] = np.random.normal(0,stdX,size=(dimX))
        noise[:,i] = np.random.normal(0,stdN,size=(dimY))

    Y = np.dot(W,X)+noise
    
    xList, yList = [], []
    for i in range(N):
        xList.append(X[:,i])
        yList.append(Y[:,i])
    xArray = np.float32(xList)
    yArray = np.float32(yList)

    return xArray, yArray

#from mutual_infomation_data.py
def generate_gaussian_data(corr, sampleSize=500, dim=20):
    mu = np.zeros(dim*2)
    cov = torch.zeros((2*dim, 2*dim))
    
    cov[torch.arange(dim), torch.arange(start=dim, end=2*dim)] = corr
    cov[torch.arange(start=dim, end=2*dim), torch.arange(dim)] = corr
    cov[torch.arange(2*dim), torch.arange(2*dim)] = 1.0

    jointSamples = np.random.multivariate_normal(mu, cov, size=sampleSize)
    xArray = jointSamples[:, 0:int(dim)]
    yArray = jointSamples[:, int(dim):int(dim*2)]
    trueMI = -0.5*np.log(np.linalg.det(cov.data))
    return xArray, yArray, trueMI

