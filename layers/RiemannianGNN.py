# Inspired by: https://github.com/facebookresearch/hgnn/blob/master/gnn/RiemannianGNN.py
# License link: https://github.com/facebookresearch/hgnn/blob/master/LICENSE

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils import *

def get_activation(args):
	if args.activation == 'leaky_relu':
		return nn.LeakyReLU(args.leaky_relu)
	elif args.activation == 'rrelu':
		return nn.RReLU()
	elif args.activation == 'relu':
		return nn.ReLU()
	elif args.activation == 'elu':
		return nn.ELU()
	elif args.activation == 'prelu':
		return nn.PReLU()
	elif args.activation == 'selu':
		return nn.SELU()

class RiemannianGNN(nn.Module):
	def __init__(self, args, manifold):
		super(RiemannianGNN, self).__init__()
		self.args = args
		self.manifold = manifold
		self.activation = get_activation(self.args)
		self.dropout = nn.Dropout(self.args.dropout)

		self.create_params()

	def create_params(self):
		# weight in euclidean space
		self.M = th.zeros([self.args.hid_dim, self.args.hid_dim], requires_grad=True)

		nn.init.xavier_uniform_(self.M)
		self.M = nn.Parameter(self.M)

	def forward(self, seq, adj, sparse=False):
		seq = th.squeeze(seq)
		seq = self.manifold.log_map_zero(seq)

		adj = th.squeeze(adj)

		combined_msg = th.mm(seq, self.M)
		combined_msg = torch.mm(adj, combined_msg)

		combined_msg = self.dropout(combined_msg)
		seq = self.manifold.exp_map_zero(combined_msg)
		seq = self.manifold.from_poincare_to_lorentz(self.activation(self.manifold.from_lorentz_to_poincare(seq)))

		return th.unsqueeze(seq, dim=0)
