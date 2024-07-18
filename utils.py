import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import scipy.linalg as linalg

def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)

    features = r_mat_inv.dot(features)
    return features

def normalized_laplacian(adj, sparse=False):
    adj = sp.coo_matrix(adj)

    d_inv_sqrt = np.power(np.array(adj.sum(1)), -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    
    if not sparse:
        return adj.todense()

    return adj.tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)

def artanh(x):
    x = x.clamp(-1 + 1e-15, 1 - 1e-15)
    z = x.double()
    return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

def mobius_add(x, y, c, dim=-1):
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(1e-15)

def poincare_sqdist(p1, p2, c):
    sqrt_c = c ** 0.5
    dist_c = artanh(
        sqrt_c * mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
    )
    dist = dist_c * 2 / sqrt_c

    return dist

def fermi_dirac_decoder(z_a, z_b):
    sqdist = poincare_sqdist(z_a, z_b, c=1.0)
    probs = 1. / (torch.exp((sqdist - 2.0) / 1.0) + 1.0)

    return probs
