import numpy as np
import torch
import torch.nn as nn


class DenseMap(nn.Module):

    def __init__(self, feat_dim=8, resolution=128, map_num=128):
        """
        feat_dim: int, feature dimension
        resolution: int, resolution of the grid
        map_num: int, number of grids
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.resolution = resolution
        self.map_num = map_num
        self.offset = resolution**2
        self.output_dim = feat_dim + 2
        map_offsets = (
            torch.arange(0, map_num * self.offset, self.offset)
            .long()
            .reshape(1, map_num, 1)
        )
        self.register_buffer("map_offsets", map_offsets)
        self.embeddings = nn.Parameter(torch.empty(map_num * self.offset, feat_dim))
        torch.nn.init.xavier_uniform_(self.embeddings)

        n_neigs = 1 << 2
        neigs = np.arange(n_neigs, dtype=np.int64).reshape((-1, 1))
        dims = np.arange(2, dtype=np.int64).reshape((1, -1))
        bin_mask = torch.tensor(neigs & (1 << dims) == 0, dtype=bool)
        self.register_buffer("bin_mask", bin_mask)

    def forward(self, inputs):
        """
        inputs: torch.Tensor, shape [batch_size, map_num, 2], in range [0, 1]
        return: torch.Tensor, shape [batch_size, map_num, feat_dim]
        """
        x = inputs * (self.resolution - 1)
        xi = x.long()
        xf = x - xi.float().detach()
        xi = xi.unsqueeze(dim=-2)
        xf = xf.unsqueeze(dim=-2)
        neigs = torch.where(self.bin_mask, xi, xi + 1)
        ids = neigs[..., 0] * self.resolution + neigs[..., 1]
        ids += self.map_offsets
        neigs_features = self.embeddings[ids]
        weights = torch.where(self.bin_mask, 1 - xf, xf)
        w = weights.prod(dim=-1, keepdim=True)
        feats = torch.sum(neigs_features * w, dim=-2)
        xf = xf.squeeze(dim=-2)
        return torch.cat([feats, xf], dim=-1)
