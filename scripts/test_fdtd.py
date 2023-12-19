import sys

sys.path.append("./")

import torch
from src.cuda_imp import FDTDSimulator, get_bound_info
from src.kleinpat_loader.model import SimpleSoundObject
from src.visualize import plot_mesh_with_plane
import os
import time
import numpy as np

import bempp.api

res = 64
freq = 200
obj = SimpleSoundObject("dataset/sphere.obj")
min_bound, max_bound, bound_size = get_bound_info(obj.vertices, padding=3.0)

fdtd = FDTDSimulator(min_bound, max_bound, bound_size, res)
vertices = torch.from_numpy(obj.vertices).cuda().float()
triangles = torch.from_numpy(obj.triangles).cuda().int()


step_num = 2000
triangles_neumann = torch.ones(
    len(triangles), step_num, dtype=torch.float32, device=vertices.device
)
for i in range(step_num):
    triangles_neumann[:, i] = torch.cos(i * fdtd.dt * 2 * torch.pi * freq)

fdtd.update(vertices, triangles, triangles_neumann)

# data = fdtd.cells[:, 1].cpu().numpy().reshape(res, res, res)
data = fdtd.accumulate_grids.reshape(res, res, res).cpu().numpy() / step_num

xs, ys, zs = fdtd.get_mgrid_xyz()

plot_mesh_with_plane(
    obj.vertices,
    obj.triangles,
    triangles_neumann[:, -1],
    xs,
    ys,
    zs,
    data,
    min_bound,
    max_bound,
    mesh_opacity=0.5,
).show()
