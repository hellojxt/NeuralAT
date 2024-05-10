import sys

sys.path.append("./")

from src.bem.solver import BEM_Solver
import matplotlib.pyplot as plt
from src.utils import Visualizer
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from time import time
import configparser
import meshio
import torch

data_dir = "dataset/NeuPAT_new/scale"

import json
import numpy as np


with open(f"{data_dir}/config.json", "r") as file:
    js = json.load(file)
    sample_config = js.get("sample", {})
    obj_config = js.get("vibration_obj", {})
    size_base = obj_config.get("size")

data = torch.load(f"{data_dir}/modal_data.pt")
vertices_base = data["vertices"]
triangles = data["triangles"]
neumann_vtx = data["neumann_vtx"]
ks_base = data["ks"]

mode_num = len(ks_base)

freq_rate = sample_config.get("freq_rate")
size_rate = sample_config.get("size_rate")
bbox_rate = sample_config.get("bbox_rate")
sample_num = sample_config.get("sample_num")
point_num_per_sample = sample_config.get("point_num_per_sample")


x = torch.zeros(sample_num, point_num_per_sample, 3 + 1 + 1, dtype=torch.float32)
# 3+1+1 is | 3: (r, theta, phi) point position | 1: size | 1: freq |
y = torch.zeros(sample_num, mode_num, point_num_per_sample, dtype=torch.float32)
# sound pressure of each mode at each point

for sample_idx in tqdm(range(sample_num)):
    freqK_base = torch.rand(1).cuda()
    freqK = freqK_base * freq_rate
    sizeK_base = torch.rand(1).cuda()
    sizeK = 1.0 / (1 + sizeK_base * (size_rate - 1))
    vertices = vertices_base * sizeK
    ks = ks_base * freqK / sizeK**0.5
    sample_points_base = torch.rand(point_num_per_sample, 3).cuda()
    rs = (sample_points_base[:, 0] * (bbox_rate - 1) + 1) * size_base * 0.7
    theta = sample_points_base[:, 1] * 2 * np.pi - np.pi
    phi = sample_points_base[:, 2] * np.pi
    xs = rs * torch.sin(phi) * torch.cos(theta)
    ys = rs * torch.sin(phi) * torch.sin(theta)
    zs = rs * torch.cos(phi)
    trg_points = torch.stack([xs, ys, zs], dim=-1)

    x[sample_idx, :, :3] = sample_points_base
    x[sample_idx, :, 3] = sizeK
    x[sample_idx, :, 4] = freqK
    bem_solver = BEM_Solver(vertices, triangles)
    for i in range(mode_num):
        dirichlet_vtx = bem_solver.neumann2dirichlet(ks[i].item(), neumann_vtx[i])
        y[sample_idx, i] = bem_solver.boundary2potential(
            ks[i].item(), neumann_vtx[i], dirichlet_vtx, trg_points
        ).abs()

        # if sample_idx == 0 and i == 0:
        #     Visualizer().add_mesh(vertices, triangles, neumann_vtx[i].abs()).add_points(
        #         trg_points, y[sample_idx, i]
        #     ).show()

torch.save({"x": x, "y": y}, f"{data_dir}/data_{sys.argv[1]}.pt")
