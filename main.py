import pickle
import random
import importlib
import argparse

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.metrics import f1_score
from sklearn.manifold import TSNE

from tqdm import tqdm
import sys

random.seed(0)
np.random.seed(0)

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

plt.rcParams["figure.figsize"] = (20, 10)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="biogrid_4503")
parser.add_argument("--hid_dim", type=int, default=96)
parser.add_argument("--activation", type=str, default="leaky_relu")
parser.add_argument('--leaky_relu', type=float, default=0.5)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--is_attn", type=bool, default=False)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--l2_coef", type=float, default=1e-05)
parser.add_argument("--n_epochs", type=int, default=1000)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--sparse", type=bool, default=False)

args = parser.parse_args()

with open("data/{}.pkl".format(args.dataset), "rb") as f:
    ds = pickle.load(f)

    features = ds["features"]
    adjs = ds["adjs"]

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_relations = len(adjs)

args.ft_size = ft_size
args.nb_relations = nb_relations

from utils import *

# Preprocess features and adjacency matrices
features = preprocess_features(features)
adjs_norm = [normalized_laplacian(adj + np.eye(nb_nodes) * 3.0, args.sparse) for adj in adjs]

# Create tensors
features = torch.FloatTensor(features[np.newaxis])

if not args.sparse:
    adjs_norm = [torch.FloatTensor(adj[np.newaxis]) for adj in adjs_norm]
else:
    adjs_norm = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs_norm]

if torch.cuda.is_available():
    features = features.cuda()
    adjs_norm = [adj.cuda() for adj in adjs_norm]

# Get labels for infomax
lbl_1 = torch.ones(nb_nodes)
lbl_0 = torch.zeros(nb_nodes)
infomax_labels = torch.cat((lbl_1, lbl_0))

if torch.cuda.is_available():
    infomax_labels = infomax_labels.cuda()

from mymodel import MyModel
from LorentzManifold import LorentzManifold

manifold = LorentzManifold(args)
model = MyModel(args, manifold)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

if torch.cuda.is_available():
    model = model.cuda()

best_loss = 1e9
best_epoch = 0
cnt_wait = 0

bce_loss = nn.BCEWithLogitsLoss()
loss_history = []

iter_embs = []

for epoch in tqdm(range(args.n_epochs)):
    model.train()
    optimizer.zero_grad()

    # Shuffle features
    idx = np.random.permutation(nb_nodes)
    fts_shuf = features[:, idx, :]
    if torch.cuda.is_available():
        fts_shuf = fts_shuf.cuda()

    logits = model(features, adjs_norm, fts_shuf, args.sparse)

    # Compute loss
    loss = bce_loss(logits.squeeze(), infomax_labels)
    loss_history.append(loss.item())

    if loss < best_loss:
        best_loss = loss.item()
        best_epoch = epoch
        cnt_wait = 0

        torch.save(model.state_dict(), "results/best_mymodel_{}.pkl".format(args.dataset))

    else:
        cnt_wait += 1

    if cnt_wait == args.patience:
        break
    
    loss.backward()
    optimizer.step()

model.load_state_dict(torch.load("results/best_mymodel_{}.pkl".format(args.dataset)))

model.eval()
with torch.no_grad():
    z = model.embed(features, adjs_norm, args.sparse, poincare=True).cpu()
    torch.save(z, "results/embs/{}.pkl".format(args.dataset))
    link_prediction_array = ds["link_prediction_array"]

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

z_a = z[0, link_prediction_array[:, 0]]
z_b = z[0, link_prediction_array[:, 1]]
labels = link_prediction_array[:, 2]

scores = []
for i in range(len(z_a)):
    scores.append(fermi_dirac_decoder(z_a[i], z_b[i]))

print(args.dataset)
print("Link prediction ROC-AUC :", roc_auc_score(labels, scores))
print("Average precision score :", average_precision_score(labels, scores))
print("")
