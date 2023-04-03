import numpy as np
import os

import scipy
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from scipy.linalg import sqrtm, logm, expm
from torch import norm
from torch.linalg import inv
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt


def np_load_files(dir_path, file_name, aggr_func = lambda x: x):
    files = [aggr_func(np.load(os.path.join(root, file)).T)
             for root, dirs, files in os.walk(dir_path)
             for file in files if(file.endswith(file_name))]
    try:
        files = np.stack(files)
    except ValueError:
        pass
    return files


def pd_load_test_files(dir_path, file_name):
    files = {root[10:]: pd.Series(np.load(os.path.join(root, file))[:, 0])
             for root, dirs, files in os.walk(dir_path)
             for file in files if(file.endswith(file_name)) if 'sub' in root}
    return pd.concat(files, axis=1)


def fnc_vec_to_sym_matrix(fnc: np.ndarray) -> np.ndarray:
    mat_dim = int((1 + np.sqrt(1 + 8 * fnc.shape[0])) / 2)
    mat = np.zeros(shape=(mat_dim, mat_dim))
    for j in range(mat_dim):
        for i in range(j):
            index = ((len(fnc) - (mat_dim-i)*(mat_dim-1-i)//2) + j - i-1)
            mat[i,j] = fnc[index]
    mat = mat + mat.T + np.identity(n=mat_dim)
    return mat


'''Data augmentations:'''


def ret_triangle_matrix(data):
    return np.array([[val for i, row in enumerate(array) for j, val in enumerate(row) if i > j] for array in data])


# For multi variate time series data for ICN: intrinsic connectivity network
def sliding_window(elements: np.ndarray, window_size: int, start_time: int) -> np.ndarray:
    if elements.shape[1] < window_size + start_time + 1:
        return np.corrcoef(x=elements[0, :], y=elements[1, :])
    return np.corrcoef(elements[:, start_time: start_time + window_size + 1])


# For FNC: functional network connectivity
def jitter(x, sigma=0.03):
    return x + np.random.normal(loc=0, scale=sigma, size=x.shape)


def sample_beta_distribution(concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = np.random.gamma(shape=[1], scale=concentration_1)
    gamma_2_sample = np.random.gamma(shape=[1], scale=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mix_up(fnc_one, fnc_two, alpha=0.2):
    images_one, labels_one = fnc_one
    images_two, labels_two = fnc_two

    la = sample_beta_distribution(alpha, alpha)
    images = images_one * la + images_two * (1 - la)
    labels = labels_one * la + labels_two * (1 - la)
    return images, labels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch_mean = lambda x: torch.from_numpy(np.mean(x, axis=2)).to(device)


def symmetric_matrix(A):
    return (A + A.T) / 2


def riemannian_mean(dFNC):
    num_matrices = dFNC.shape[2]
    if num_matrices == 1:
        return dFNC
    M = torch_mean(dFNC)
    weights = (torch.ones((num_matrices, 1)) / num_matrices).to(device)
    for iteration in range(30):
        A = torch.from_numpy(sqrtm(M.cpu())).to(device)
        B = inv(A).to(device)
        S = torch.zeros_like(M).to(device)
        for j in range(num_matrices):
            current_matrix = torch.from_numpy(dFNC[:, :, j]).to(device)
            if scipy.linalg.eig(current_matrix.cpu())[0].min() <= 0:
                continue
            BCB = symmetric_matrix(B @ current_matrix @ B)
            S += weights[j] * torch.from_numpy(logm(BCB.cpu())).to(device)
        M = symmetric_matrix(A @ torch.from_numpy(expm(S.cpu())).to(device) @ A)
        if norm(S, 'fro') < 1e-6:
            break
    return M.cpu()


def conjection(dFNC):
    x_bar = riemannian_mean(dFNC)
    E = scipy.linalg.fractional_matrix_power(x_bar, -0.5)
    gpu_dFNC = torch.from_numpy(dFNC.transpose(2, 0, 1)).to(device)
    gpu_E = torch.from_numpy(E.T).to(device)
    temp = torch.matmul(gpu_dFNC, gpu_E)
    x_hat = torch.matmul(gpu_E, temp)
    return x_hat.cpu()

