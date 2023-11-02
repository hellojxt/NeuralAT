import sys

sys.path.append("./")

import torch
from src.cuda_imp import Sampler
from src.bem.bempp import BEMModel
from src.loader.model import ModalSoundObject
import numpy as np
from src.net import get_mlps
from src.visualize import plot_mesh

# Sample vertices (3D coordinates of vertices)
vertices = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
).cuda()

# Sample triangles (indices of vertices forming triangles)
triangles = torch.tensor(
    [
        [0, 1, 2],
        [0, 2, 3],
    ],
    dtype=torch.int32,
).cuda()

# Sample triangle_neumann (Neumann values for each triangle)
triangle_neumann = torch.tensor([0.1, 0.2], dtype=torch.float32).cuda()

# Sample triangle_importance (optional, importance values for each triangle)
triangle_importance = torch.tensor([1.0, 1.0], dtype=torch.float32).cuda()

# Create a Sampler instance
sampler = Sampler(
    vertices, triangles, triangle_neumann, triangle_importance, alias_factor=8
)

k = 1.0
# sampler.sample_points(k, candidate_num=32, reuse_num=4)
sampler.sample_points(k, candidate_num=128, reuse_num=4)
sampler.print_point_pairs()
