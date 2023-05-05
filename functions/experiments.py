from functions.algorithms import DKRR, DCNY, DNystroem, RFKRR, Nystroem, DCRF
from functions.utils import load_data, error_estimate
from functions.optimal_parameters import get_parameter
import pickle
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def exp1_landmarks(dataset_name, m, M_arr, repeat = 10, device = 'cpu'):
    logger = logging.getLogger('Exp1 difference between DKRR, DCNY and DNystroem w.r.t. the number of landmarks')
    logger.setLevel(logging.INFO)
    torch.manual_seed(1)
    np.random.seed(1)

    parameter_dic = get_parameter(dataset_name)
    task_type = parameter_dic['task_type']
    num_classes = parameter_dic['num_classes']
    d = parameter_dic['dimensional']
    k_par= parameter_dic['kernel_par']
    lambda_reg = parameter_dic['lambda_reg']

    n = parameter_dic['num_examples']
    data_dir = './data'
    result_dir = './results'
    mse_mat = np.zeros((3, len(M_arr), repeat))
    error_mat = np.zeros((3, len(M_arr), repeat))
    time_mat = np.zeros((3, len(M_arr), repeat))

    for j in range(repeat):
        trainloader, testloader, d, num_classes = load_data(dataset_name, n)
        X_train, y_train = next(iter(trainloader))
        X_test, y_test = next(iter(testloader))
        X_train, y_train, X_test, y_test = X_train.to(device=device, dtype=torch.float64), torch.nn.functional.one_hot(y_train, num_classes).to(device=device, dtype=torch.float64), X_test.to(device=device, dtype=torch.float64), y_test.to(device=device, dtype=torch.float64)

        start = time.time()
        y_pred = DKRR(X_train, y_train, X_test, m,  k_par, lambda_reg, device = device)
        time_mat[0, :, j] = time.time() - start
        mse_mat[0, :, j], error_mat[0, :, j] = error_estimate(y_pred, y_test, task_type)

        for (i, M) in enumerate(M_arr):
            landmarks = X_train[np.random.choice(n, M, replace = False), :]

            if M <=  int(n/m):
                start = time.time()
                y_pred = DCNY(X_train, y_train, X_test, M, m,  k_par, lambda_reg, device = device)
                time_mat[1, i, j] = time.time() - start
                mse_mat[1, i, j], error_mat[1, i, j] = error_estimate(y_pred, y_test, task_type)
            else:
                time_mat[1, i, j], mse_mat[1, i, j], error_mat[1, i, j] = time_mat[1, i-1, j], mse_mat[1, i-1, j], error_mat[1, i-1, j]

            start = time.time()
            y_pred = DNystroem(X_train, y_train, X_test, landmarks, m,  k_par, lambda_reg, device = device)
            time_mat[2, i, j] = time.time() - start
            mse_mat[2, i, j], error_mat[2, i, j] = error_estimate(y_pred, y_test, task_type)
            
            logger.info("Exp 1 --- # Round {}, # Repeat {}".format(i, j))
    
    data_ = {
        'partitions': m,
        'landmarks' : M_arr,
        'error': error_mat,
        'mse' : mse_mat,
        'time' : time_mat,
        'name': ['DKRR', 'DC-NY', 'DNystroem']
    }
    result_path = '{}/exp1_landmarks_{}.pkl'.format(result_dir, dataset_name)
    with open(result_path, "wb") as f:
        pickle.dump(data_, f)

def exp1_examples(dataset_name, m, M, n_arr, repeat = 10, device = 'cpu'):
    logger = logging.getLogger('Exp1 difference between DKRR, DCNY and DNystroem w.r.t. the sample size')
    logger.setLevel(logging.INFO)
    torch.manual_seed(1)
    np.random.seed(1)

    parameter_dic = get_parameter(dataset_name)
    task_type = parameter_dic['task_type']
    num_classes = parameter_dic['num_classes']
    d = parameter_dic['dimensional']
    k_par= parameter_dic['kernel_par']
    lambda_reg = parameter_dic['lambda_reg']

    data_dir = './data'
    result_dir = './results'
    mse_mat = np.zeros((3, len(n_arr), repeat))
    error_mat = np.zeros((3, len(n_arr), repeat))
    time_mat = np.zeros((3, len(n_arr), repeat))

    for j in range(repeat):
        trainloader, testloader, d, num_classes = load_data(dataset_name, parameter_dic['num_examples'])
        X_train, y_train = next(iter(trainloader))
        X_test, y_test = next(iter(testloader))
        X_train_all, y_train_all, X_test, y_test = X_train.to(device=device, dtype=torch.float64), torch.nn.functional.one_hot(y_train, num_classes).to(device=device, dtype=torch.float64), X_test.to(device=device, dtype=torch.float64), y_test.to(device=device, dtype=torch.float64)

        for (i, n) in enumerate(n_arr):
            X_train, y_train =  X_train_all[0:n, :], y_train_all[0:n]

            start = time.time()
            y_pred = DKRR(X_train, y_train, X_test, m,  k_par, lambda_reg, device = device)
            time_mat[0, i, j] = time.time() - start
            mse_mat[0, i, j], error_mat[0, i, j] = error_estimate(y_pred, y_test, task_type)

            start = time.time()
            y_pred = DCNY(X_train, y_train, X_test, M, m,  k_par, lambda_reg, device = device)
            time_mat[1, i, j] = time.time() - start
            mse_mat[1, i, j], error_mat[1, i, j] = error_estimate(y_pred, y_test, task_type)

            if M > n:
                M = n
            landmarks = X_train[np.random.choice(n, M, replace = False), :]
            start = time.time()
            y_pred = DNystroem(X_train, y_train, X_test, landmarks, m,  k_par, lambda_reg, device = device)
            time_mat[2, i, j] = time.time() - start
            mse_mat[2, i, j], error_mat[2, i, j] = error_estimate(y_pred, y_test, task_type)
            
            logger.info("Exp 1 --- # Round {}, # Repeat {}".format(i, j))
    
    data_ = {
        'partitions': m,
        'landmarks' : M,
        'sample_size' : n_arr,
        'error': error_mat,
        'mse' : mse_mat,
        'time' : time_mat,
        'name': ['DKRR', 'DC-NY', 'DNystroem']
    }
    result_path = '{}/exp1_examples_{}.pkl'.format(result_dir, dataset_name)
    with open(result_path, "wb") as f:
        pickle.dump(data_, f)

def exp2_partitions(dataset_name, m_arr, M, repeat = 10, device = 'cpu'):
    logger = logging.getLogger('Exp1 difference between DKRR, DCNY and DNystroem w.r.t. the number of partitions')
    logger.setLevel(logging.INFO)
    torch.manual_seed(1)
    np.random.seed(1)

    parameter_dic = get_parameter(dataset_name)
    task_type = parameter_dic['task_type']
    num_classes = parameter_dic['num_classes']
    d = parameter_dic['dimensional']
    k_par= parameter_dic['kernel_par']
    lambda_reg = parameter_dic['lambda_reg']

    n = parameter_dic['num_examples']
    data_dir = './data'
    result_dir = './results'
    mse_mat = np.zeros((6, len(m_arr), repeat))
    error_mat = np.zeros((6, len(m_arr), repeat))
    time_mat = np.zeros((6, len(m_arr), repeat))

    for j in range(repeat):
        trainloader, testloader, d, num_classes = load_data(dataset_name, parameter_dic['num_examples'])
        X_train, y_train = next(iter(trainloader))
        X_test, y_test = next(iter(testloader))
        X_train, y_train, X_test, y_test = X_train.to(device=device, dtype=torch.float64), torch.nn.functional.one_hot(y_train, num_classes).to(device=device, dtype=torch.float64), X_test.to(device=device, dtype=torch.float64), y_test.to(device=device, dtype=torch.float64)

        landmarks = X_train[np.random.choice(n, M, replace = False), :]
        
        start = time.time()
        y_pred = RFKRR(X_train, y_train, X_test, M, k_par, lambda_reg, device = device)
        time_mat[0, :, j] = time.time() - start
        mse_mat[0, :, j], error_mat[0, :, j] = error_estimate(y_pred, y_test, task_type)

        start = time.time()
        y_pred = Nystroem(X_train, y_train, X_test, landmarks, k_par, lambda_reg, device = device)
        time_mat[1, :, j] = time.time() - start
        mse_mat[1, :, j], error_mat[1, :, j] = error_estimate(y_pred, y_test, task_type)

        for (i, m) in enumerate(m_arr):

            start = time.time()
            y_pred = DKRR(X_train, y_train, X_test, m,  k_par, lambda_reg, device = device)
            time_mat[2, i, j] = time.time() - start
            mse_mat[2, i, j], error_mat[2, i, j] = error_estimate(y_pred, y_test, task_type)

            start = time.time()
            y_pred = DCNY(X_train, y_train, X_test, M, m,  k_par, lambda_reg, device = device)
            time_mat[3, i, j] = time.time() - start
            mse_mat[3, i, j], error_mat[3, i, j] = error_estimate(y_pred, y_test, task_type)
            
            start = time.time()
            y_pred = DCRF(X_train, y_train, X_test, m, M, k_par, lambda_reg, device = device)
            time_mat[4, i, j] = time.time() - start
            mse_mat[4, i, j], error_mat[4, i, j] = error_estimate(y_pred, y_test, task_type)


            start = time.time()
            y_pred = DNystroem(X_train, y_train, X_test, landmarks, m,  k_par, lambda_reg, device = device)
            time_mat[5, i, j] = time.time() - start
            mse_mat[5, i, j], error_mat[5, i, j] = error_estimate(y_pred, y_test, task_type)
            
            logger.info("Exp 1 --- # Round {}, # Repeat {}".format(i, j))
    
    data_ = {
        'partitions': m_arr,
        'landmarks' : M,
        'error': error_mat,
        'mse' : mse_mat,
        'time' : time_mat,
        'name': ['RF', 'Nystroem', 'DKRR', 'DC-NY', 'DCRF', 'DNystroem']
    }
    result_path = '{}/exp2_partitions_{}.pkl'.format(result_dir, dataset_name)
    with open(result_path, "wb") as f:
        pickle.dump(data_, f)

def exp3_landmarks(dataset_name, m, M_arr, repeat = 10, device = 'cpu'):
    logger = logging.getLogger('Exp1 difference w.r.t. the number of landmarks')
    logger.setLevel(logging.INFO)
    torch.manual_seed(1)
    np.random.seed(1)

    parameter_dic = get_parameter(dataset_name)
    task_type = parameter_dic['task_type']
    num_classes = parameter_dic['num_classes']
    d = parameter_dic['dimensional']
    k_par= parameter_dic['kernel_par']
    lambda_reg = parameter_dic['lambda_reg']

    n = parameter_dic['num_examples']
    data_dir = './data'
    result_dir = './results'
    mse_mat = np.zeros((6, len(M_arr), repeat))
    error_mat = np.zeros((6, len(M_arr), repeat))
    time_mat = np.zeros((6, len(M_arr), repeat))

    for j in range(repeat):
        trainloader, testloader, d, num_classes = load_data(dataset_name, n)
        X_train, y_train = next(iter(trainloader))
        X_test, y_test = next(iter(testloader))
        X_train, y_train, X_test, y_test = X_train.to(device=device, dtype=torch.float64), torch.nn.functional.one_hot(y_train, num_classes).to(device=device, dtype=torch.float64), X_test.to(device=device, dtype=torch.float64), y_test.to(device=device, dtype=torch.float64)

        start = time.time()
        y_pred = DKRR(X_train, y_train, X_test, m,  k_par, lambda_reg, device = device)
        time_mat[0, :, j] = time.time() - start
        mse_mat[0, :, j], error_mat[0, :, j] = error_estimate(y_pred, y_test, task_type)

        for (i, M) in enumerate(M_arr):
            landmarks = X_train[np.random.choice(n, M, replace = False), :]
            
            start = time.time()
            y_pred = RFKRR(X_train, y_train, X_test, M, k_par, lambda_reg, device = device)
            time_mat[1, i, j] = time.time() - start
            mse_mat[1, i, j], error_mat[1, i, j] = error_estimate(y_pred, y_test, task_type)

            start = time.time()
            y_pred = Nystroem(X_train, y_train, X_test, landmarks, k_par, lambda_reg, device = device)
            time_mat[2, i, j] = time.time() - start
            mse_mat[2, i, j], error_mat[2, i, j] = error_estimate(y_pred, y_test, task_type)

            if M <=  int(n/m):
                start = time.time()
                y_pred = DCNY(X_train, y_train, X_test, M, m,  k_par, lambda_reg, device = device)
                time_mat[3, i, j] = time.time() - start
                mse_mat[3, i, j], error_mat[3, i, j] = error_estimate(y_pred, y_test, task_type)
            else:
                time_mat[3, i, j], mse_mat[3, i, j], error_mat[3, i, j] = time_mat[3, i-1, j], mse_mat[3, i-1, j], error_mat[3, i-1, j]

            start = time.time()
            y_pred = DCRF(X_train, y_train, X_test, m, M, k_par, lambda_reg, device = device)
            time_mat[4, i, j] = time.time() - start
            mse_mat[4, i, j], error_mat[4, i, j] = error_estimate(y_pred, y_test, task_type)

            start = time.time()
            y_pred = DNystroem(X_train, y_train, X_test, landmarks, m,  k_par, lambda_reg, device = device)
            time_mat[5, i, j] = time.time() - start
            mse_mat[5, i, j], error_mat[5, i, j] = error_estimate(y_pred, y_test, task_type)
            
            logger.info("Exp 3 --- # Round {}, # Repeat {}".format(i, j))
    
    data_ = {
        'partitions': m,
        'landmarks' : M_arr,
        'error': error_mat,
        'mse' : mse_mat,
        'time' : time_mat,        
        'name': ['DKRR', 'RF', 'Nystroem', 'DC-NY', 'DCRF', 'DNystroem']
    }
    result_path = '{}/exp3_landmarks_{}.pkl'.format(result_dir, dataset_name)
    with open(result_path, "wb") as f:
        pickle.dump(data_, f)

