import torch
import torch.nn as nn

from layers import HierarchicalAggrLayer, GCN, DiscriminatorDGI, RiemannianGNN

class MyModel(nn.Module):
    def __init__(self, args, manifold):
        super().__init__()
        self.args = args
        self.hid_dim = args.hid_dim
        self.manifold = manifold
        
        self.pre_gcn = GCN(args.ft_size, args.hid_dim, drop_prob=args.dropout)
        self.hier_layer = HierarchicalAggrLayer(args, manifold, args.nb_relations, 1)
        
        self.hgcn = RiemannianGNN(args, manifold)
        self.disc = DiscriminatorDGI(args.hid_dim)

    def forward(self, fts, adjs_norm, fts_shuf, sparse=False):
        z_pos = self.embed(fts, adjs_norm, sparse)
        z_neg = self.embed(fts_shuf, adjs_norm, sparse)

        s = torch.mean(z_pos, dim=1)
        s = torch.sigmoid(s)
        s = torch.unsqueeze(s, dim=1)

        logits = self.disc(s, z_pos, z_neg)
        return logits

    def embed(self, fts, adjs_norm, sparse=False, poincare=False):
        hid_fts = 0
        for adj in adjs_norm:
            hid_fts += self.pre_gcn(fts, adj, sparse)
        hid_fts /= len(adjs_norm)

        z, new_adjs = self.hier_layer(hid_fts, adjs_norm, sparse)
        z = self.hgcn(z, new_adjs[0], sparse)

        if poincare:
            z = self.manifold.from_lorentz_to_poincare(z)

        return z
