"""
Algorithms for sovling KRR problems
"""

import torch
import os
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

def gaussian_kernel(X1, X2, sigma):
    D = torch.square(torch.norm(X1, 2, dim=1)).repeat(X2.shape[0], 1).T - 2*X1.mm(X2.T) + torch.square(torch.norm(X2, 2, dim=1)).repeat(X1.shape[0], 1)
    return torch.exp(- D/ (2 * sigma * sigma))

# forward feature mapping and load to device
def RFF(X_train, X_test, k_par = 10, D = 200, device = 'cpu'):
    d = X_train.shape[1]
    m = Uniform(torch.tensor([0.0], device=device), torch.tensor([2 * torch.pi], device=device))
    W = torch.normal(0, 1 / (k_par*k_par), size=(d, D), dtype=torch.float64, device = device)
    b = m.sample((1, D)).view(-1, D).to(device)
    X_train_FM =1 / np.sqrt(D) * torch.cos(torch.matmul(X_train, W) + b)
    X_test_FM = 1 / np.sqrt(D) * torch.cos(torch.matmul(X_test, W) + b)
    return X_train_FM, X_test_FM

def KRR(X_train, y_train, X_test, sigma = 1, lambda_reg = 0.01, device = 'cpu'):
    K = gaussian_kernel(X_train, X_train, sigma)
    n = X_train.shape[0]
    alpha = torch.linalg.pinv(K + lambda_reg * n * torch.eye(n, device=device)).mm(y_train.to(torch.float64))
    return gaussian_kernel(X_test, X_train, sigma).mm(alpha)

def Nystroem(X_train, y_train, X_test, landmarks, sigma = 1, lambda_reg = 0.01, device = 'cpu'):
    KnM = gaussian_kernel(X_train, landmarks, sigma)
    KMM = gaussian_kernel(landmarks, landmarks, sigma)
    n = X_train.shape[0]
    M = landmarks.shape[0]
    alpha = torch.linalg.pinv(KnM.T.mm(KnM) + lambda_reg * n * KMM).mm(KnM.T).mm(y_train.to(torch.float64))
    return gaussian_kernel(X_test, landmarks, sigma).mm(alpha)

def DKRR(X_train, y_train, X_test, m = 10, sigma = 1, lambda_reg = 0.01, device = 'cpu'):
    n = X_train.shape[0]
    nts = X_test.shape[0]
    step = int(n/m)
    ypred = torch.zeros(nts, y_train.shape[1], device=device)
    for k in range(m):
        idx_k = range(step*k, step*(k+1))
        part_X = X_train[idx_k, :]
        part_y = y_train[idx_k]
        K = gaussian_kernel(part_X, part_X, sigma)
        alpha = torch.linalg.pinv(K + lambda_reg * n * torch.eye(step, device=device)).mm(part_y.to(torch.float64))
        ypred += gaussian_kernel(X_test, part_X, sigma).mm(alpha) / m
    return ypred

def RFKRR(X_train, y_train, X_test, D = 200, k_par = 1, lambda_reg = 0.01, device = 'cpu'):
    X_train, X_test = RFF(X_train, X_test, k_par, D, device=device)
    W = torch.linalg.inv(X_train.T.mm(X_train) + lambda_reg * torch.eye(D, device=device) * X_train.shape[0]).mm(X_train.T).mm(y_train.to(torch.float64))
    return X_test.mm(W)
    
def DCRF(X_train, y_train, X_test, m = 10, D = 200, k_par = 1, lambda_reg = 0.01, device = 'cpu'):
    n = X_train.shape[0]
    step = int(n/m)
    W = torch.zeros(D, y_train.shape[1], dtype=torch.float64, device=device)
    X_train, X_test = RFF(X_train, X_test, k_par, D, device=device)
    for k in range(m):
        idx_k = range(step*k, step*(k+1))
        part_X = X_train[idx_k, :]
        part_y = y_train[idx_k]
        W += torch.linalg.inv(part_X.T.mm(part_X) + lambda_reg * torch.eye(D, device=device) * part_X.shape[0]).mm(part_X.T).mm(part_y.to(torch.float64))
    return X_test.mm(W)


def DCNY(X_train, y_train, X_test, M, m = 10, sigma = 1, lambda_reg = 0.01, device = 'cpu'):
    n = X_train.shape[0]
    nts = X_test.shape[0]
    step = int(n/m)
    ypred = torch.zeros(nts, y_train.shape[1], device=device)
    for k in range(m):
        idx_k = range(step*k, step*(k+1))
        part_X = X_train[idx_k, :]
        part_y = y_train[idx_k]
        if M < len(part_y):
            landmarks = part_X[np.random.choice(len(part_y), M, replace=False), :]
        else:
            landmarks = part_X
        ypred += Nystroem(part_X, part_y, X_test, landmarks, sigma, lambda_reg) / m
    return ypred

def DNystroem(X_train, y_train, X_test, landmarks, m = 10, sigma = 1, lambda_reg = 0.01, device = 'cpu'):
    n = X_train.shape[0]
    nts = X_test.shape[0]
    step = int(n/m)
    ypred = torch.zeros(nts, y_train.shape[1], device=device)
    for k in range(m):
        part_X = X_train[step*k : step*(k+1), :]
        part_y = y_train[step*k : step*(k+1)]
        ypred += Nystroem(part_X, part_y, X_test, landmarks, sigma, lambda_reg) / m
    return ypred


def recursive_RLS(X, M, dataset, lambda_reg = 1e-6, sigma = 1, device = 'cpu'):
    n = X.shape[0]
    if os.path.exists('./data/{}_RLS.pkl'.format(dataset)):
        with open('./data/{}_RLS.pkl'.format(dataset), 'rb') as f:
            scores = pickle.load(f)
    else:
        landmarks_M = np.random.choice(n, M, replace = False)
        KS = gaussian_kernel(X, X[landmarks_M, :], sigma)
        SS = gaussian_kernel(X[landmarks_M, :], X[landmarks_M, :], sigma)
        scores = torch.diag(KS.mm(torch.linalg.pinv(SS + lambda_reg * n * torch.eye(M, device = device)).mm(KS.T))) / (lambda_reg * n)

    return scores