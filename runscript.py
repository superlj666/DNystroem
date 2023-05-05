from functions.experiments import exp1_landmarks, exp1_examples, exp2_partitions, exp3_landmarks
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#### Exp1
M_arr = np.linspace(100, 2000, 20).astype(int)
exp1_landmarks('mnist', 60, M_arr, 10, device = device)

n_arr = np.linspace(1000, 60000, 20).astype(int)
exp1_examples('mnist', 60, 500, n_arr, 10, device = device)

# #### Exp2
m_arr = np.linspace(10, 50, 20).astype(int)
exp2_partitions('usps', m_arr, 800, 10, device = device)

m_arr = np.linspace(10, 50, 20).astype(int)
exp2_partitions('pendigits', m_arr, 400, 10, device = device)

m_arr = np.linspace(10, 50, 20).astype(int)
exp2_partitions('letter', m_arr, 800, 10, device = device)

m_arr = np.linspace(30, 300, 20).astype(int)
exp2_partitions('mnist', m_arr, 500, 10, device = device)

#### Exp3
M_arr = (np.logspace(-1, 0, base=20, num=10) * 1000).astype(int)
exp3_landmarks('usps', 20, M_arr, 10, device = device)

M_arr = (np.logspace(-1, 0, base=20, num=20) * 1000).astype(int)
exp3_landmarks('pendigits', 20, M_arr, 10, device = device)

M_arr = (np.logspace(-1, 0, base=20, num=10) * 1000).astype(int)
exp3_landmarks('letter', 30, M_arr, 10, device = device)

M_arr = (np.logspace(-1, 0, base=20, num=10) * 2000).astype(int)
exp3_landmarks('mnist', 60, M_arr, 10, device = device) 