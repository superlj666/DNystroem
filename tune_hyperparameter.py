"""
Tune Hyperparamters for KRR via NNI.
"""

from functions.algorithms import KRR, Nystroem, DKRR, RFKRR, DCRF, DCNY, DNystroem
from functions.utils import load_data, error_estimate
from functions.optimal_parameters import get_parameter
import os
import pickle
import argparse
import logging
import nni
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nni.utils import merge_parameter
from torchvision import datasets, transforms

CUDA_LAUNCH_BLOCKING=1
SAVE = False
logger = logging.getLogger('Tune Hyperparamters')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64) 

def main(args):
    torch.manual_seed(args['seed']) 
    np.random.seed(args['seed']) 

    dataset_name = args['dataset']
    estimator = args['estimator']
    m = args['m']
    M = args['M']

    parameter_dic = get_parameter(dataset_name)
    task_type = parameter_dic['task_type']
    num_classes = parameter_dic['num_classes']    
    n = parameter_dic['num_examples']

    k_par = args['kernel_par'] # parameter_dic['kernel_par'] #
    lambda_reg = args['lambda_reg'] # parameter_dic['lambda_reg'] #

    trainloader, testloader, d, num_classes = load_data(dataset_name, n)
    X_train, y_train = next(iter(trainloader))
    X_test, y_test = next(iter(testloader))
    X_train, y_train, X_test, y_test = X_train.to(device=device, dtype=torch.float64), torch.nn.functional.one_hot(y_train, num_classes).to(device=device, dtype=torch.float64), X_test.to(device=device, dtype=torch.float64), y_test.to(device=device, dtype=torch.float64)
    landmarks = X_train[np.random.choice(n, M, replace = False), :]

    if estimator == 'KRR':
        y_pred = KRR(X_train, y_train, X_test, k_par, lambda_reg, device = device)
        mse, error_rate = error_estimate(y_pred, y_test, task_type)
    elif estimator == 'Nystroem':
        y_pred = Nystroem(X_train, y_train, X_test, landmarks,  k_par, lambda_reg, device = device)
        mse, error_rate = error_estimate(y_pred, y_test, task_type)
    elif estimator == 'DKRR':
        y_pred = DKRR(X_train, y_train, X_test, m,  k_par, lambda_reg, device = device)
        mse, error_rate = error_estimate(y_pred, y_test, task_type)
    elif estimator == 'RFKRR':
        y_pred = RFKRR(X_train, y_train, X_test, M,  k_par, lambda_reg, device = device)
        mse, error_rate = error_estimate(y_pred, y_test, task_type)
    elif estimator == 'DCRF':
        y_pred = DCRF(X_train, y_train, X_test, m, M,  k_par, lambda_reg, device = device)
        mse, error_rate = error_estimate(y_pred, y_test, task_type)
    elif estimator == 'DCNY':
        y_pred = DCNY(X_train, y_train, X_test, M, m,  k_par,  lambda_reg, device = device)
        mse, error_rate = error_estimate(y_pred, y_test, task_type)
    elif estimator == 'DNystroem':
        y_pred = DNystroem(X_train, y_train, X_test, landmarks, m,  k_par, lambda_reg, device = device)
        mse, error_rate = error_estimate(y_pred, y_test, task_type)
        logger.info("DNystroem --- lambda: {}, sigma: {}, Error: {:.5f}".format(lambda_reg, k_par, error_rate))
    
    nni.report_final_result(error_rate)
    logger.debug('Final result is %.4f', error_rate)
    logger.debug('Send final result done.')

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='KRR Tuner')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument("--estimator", type=str, default='DNystroem', help="estimator name")
    parser.add_argument("--dataset", type=str, default='EEG', help="dataset name")
    parser.add_argument("--kernel_par", type=float, default=10.01, help="kernel hyperparameter")
    parser.add_argument('--lambda_reg', type=float, default=0.00001, help='regularizer parameter (default: 0.01)')
    parser.add_argument('--m', type=int, default=30, help='the number of partitions')
    parser.add_argument('--M', type=int, default=500, help='the number of Nystroem landmarks')
    parser.add_argument("--data_dir", type=str,
                        default='./data', help="data directory")
    args, _ = parser.parse_known_args()
    return args

## set parameters in config_gpu.yml 
## and run nnictl create --config /home/superlj666/Experiment/FedNewton/config_gpu.yml
if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)

    except Exception as exception:
        logger.exception(exception)
        raise