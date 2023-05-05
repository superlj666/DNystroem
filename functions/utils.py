import os
import numpy as np
from math import ceil
from random import Random
import scipy.io
import pickle

import torch
import torch.utils.data as data_utils
from torch import float32
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision.models as IMG_models
from sklearn.datasets import load_svmlight_file
from torch.utils.data import Dataset
import json
# from models import *

class Logger(object):
    def __init__(self,filename):
        self.log=open(filename,'w')
    def write(self,content):
        self.log.write(content)
        self.log.flush()

def is_regression(dataest):
    if dataest == 'abalone' or dataest == 'abalone' or dataest == 'cadata' or dataest == 'cpusmall' or dataest == 'space_ga' :
        return True

class SVMLightDataset(Dataset):
    def __init__(self, data_name, outputs, inputs, transform=None, target_transform=None):
        self.outputs = outputs
        self.inputs = inputs
        if is_regression(data_name):
            self.outputs = 100*(self.outputs - self.outputs.min()) / (self.outputs.max() - self.outputs.min())
        else:
            if len(set(self.outputs)) == 2:
                self.outputs = (self.outputs - self.outputs.min()) / (self.outputs.max() - self.outputs.min())
            elif len(set(self.outputs))  > 2:
                self.outputs -= self.outputs.min()
        # self.inputs = self.inputs.toarray()
        # self.outputs = self.outputs.toarray()
        self.transform = transform
        self.target_transform = target_transform
        self.data_name = data_name

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        sample = torch.tensor(self.inputs[idx].A, dtype=torch.float64).view(-1)
        if is_regression(self.data_name):
            label = torch.tensor(self.outputs[idx], dtype=torch.float64)
        else:
            label = torch.tensor(self.outputs[idx], dtype=torch.long)
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)
        return sample, label

class ArrayDataset(Dataset):
    def __init__(self, data_name, outputs, inputs, transform=None, target_transform=None):
        self.outputs = outputs
        self.inputs = inputs
        if is_regression(data_name):
            self.outputs = 100*(self.outputs - self.outputs.min()) / (self.outputs.max() - self.outputs.min())
        else:
            if len(set(self.outputs)) == 2:
                self.outputs = (self.outputs - self.outputs.min()) / (self.outputs.max() - self.outputs.min())
            elif len(set(self.outputs))  > 2:
                self.outputs -= self.outputs.min()
        # self.inputs = self.inputs.toarray()
        # self.outputs = self.outputs.toarray()
        self.transform = transform
        self.target_transform = target_transform
        self.data_name = data_name

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        sample = torch.tensor(self.inputs[idx], dtype=torch.float64).view(-1)
        if is_regression(self.data_name):
            label = torch.tensor(self.outputs[idx], dtype=torch.float64)
        else:
            label = torch.tensor(self.outputs[idx], dtype=torch.long)
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)
        return sample, label


def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 标准化处理
    x = x.reshape((-1,)) # 拉平
    x = torch.tensor(x)
    return x

def load_data(dataset_name, batch_size = 32):
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=data_tf)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=data_tf)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                                 shuffle=False)
        return trainloader, testloader, 3072, 10
    elif dataset_name == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=data_tf)
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                               download=True, transform=data_tf)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                                 shuffle=False)
        return trainloader, testloader, 784, 10
    elif dataset_name == 'EEG':
        mat = scipy.io.loadmat('./datasets/EEG.mat')
        trainset = ArrayDataset('EEG', mat['y'].reshape(-1), mat['X'])
        feature_size = trainset.inputs.shape[1]
        class_size = len(set(trainset.outputs))
        trainset, testset = torch.utils.data.random_split(trainset, [int(len(trainset)*0.8), len(trainset) - int(len(trainset)*0.8)])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
        return trainloader, testloader, feature_size, class_size
    else:
        inputs, outputs = load_svmlight_file('./datasets/' + dataset_name)
        trainset = SVMLightDataset(dataset_name, outputs, inputs)
        feature_size = trainset.inputs.shape[1]
        class_size = len(set(trainset.outputs))
        if os.path.exists('./datasets/' + dataset_name + '.t'):
            inputs, outputs = load_svmlight_file('./datasets/' + dataset_name + '.t')
            testset = SVMLightDataset(dataset_name, outputs, inputs)
        else:
            trainset, testset = torch.utils.data.random_split(trainset, [int(len(trainset)*0.8), len(trainset) - int(len(trainset)*0.8)])
        if dataset_name == 'a7a':
            testset.inputs = testset.inputs[:, :122]
        elif dataset_name == 'a8a':
            trainset.inputs = trainset.inputs[:, :122]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
        return trainloader, testloader, feature_size, 1 if is_regression(dataset_name) else class_size

def error_estimate(output, target, task_type = 'regression'):
    mse_loss = nn.MSELoss(reduction='mean')
    if task_type == 'binary':
        topK = comp_accuracy(output, target)
        target = torch.nn.functional.one_hot(target.to(torch.long), output.shape[-1])
        mse = mse_loss(output, target)
        return mse.item(), 1 - topK[0].item() / 100
    elif task_type == 'multiclass':
        topK = comp_accuracy(output, target)
        target = torch.nn.functional.one_hot(target.to(torch.long), output.shape[-1])
        mse = mse_loss(output, target)
        return mse.item(), 1 - topK[0].item() / 100
    elif task_type == 'regression':
        return mse_loss(output, target).item(), mse_loss(output, target).item()
    else:
        print("Unsupport task type: {}".format(task_type))


def comp_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res