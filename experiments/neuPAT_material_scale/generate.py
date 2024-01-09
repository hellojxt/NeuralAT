import sys

sys.path.append("./")

from src.modalsound.model import (
    ModalSoundObj,
    MatSet,
    Material,
    BEMModel,
    MeshObj,
    get_spherical_surface_points,
)
from src.cuda_imp import (
    ImportanceSampler,
    MonteCarloWeight,
    get_weights_boundary_ks_base,
    get_weights_potential_ks_base,
)
import matplotlib.pyplot as plt
from src.ffat_solve import monte_carlo_solve, bem_solve
from src.visualize import plot_point_cloud, plot_mesh, CombinedFig
from src.solver import BiCGSTAB_batch
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from time import time
import configparser
import meshio
import torch

data_dir = "dataset/NeuPAT/scale"

import json
import numpy as np


with open(f"{data_dir}/config.json", "r") as file:
    config_data = json.load(file)

data = torch.load(f"{data_dir}/modal_data.pt")
vertices_base = data["vertices"]
triangles = data["triangles"]
neumann_tri = data["neumann_tri"]
ks_base = data["ks"]
mode_num = len(ks_base)

trg_pos_min = torch.tensor(
    config_data.get("solver", {}).get("trg_pos_min"), device="cuda", dtype=torch.float32
)
trg_pos_max = torch.tensor(
    config_data.get("solver", {}).get("trg_pos_max"), device="cuda", dtype=torch.float32
)
size_min = np.log(config_data.get("solver", {}).get("size_scale_min"))
size_max = np.log(config_data.get("solver", {}).get("size_scale_max"))
freq_min = np.log(config_data.get("solver", {}).get("freq_scale_min"))
freq_max = np.log(config_data.get("solver", {}).get("freq_scale_max"))
print("trg_pos_min:", trg_pos_min)
print("trg_pos_max:", trg_pos_max)
trg_sample_num = config_data.get("solver", {}).get("trg_sample_num", 1000)
src_sample_num = config_data.get("solver", {}).get("src_sample_num", 1000)
print("trg_sample_num:", trg_sample_num)
print("src_sample_num:", src_sample_num)


def monte_carlo_process(vertices, ks, trg_points):
    return monte_carlo_solve(vertices, triangles, neumann_tri, ks, trg_points, 5000)


def bem_process(vertices, ks, trg_points):
    return bem_solve(vertices, triangles, neumann_tri, ks, trg_points)


check_correct = False

x = torch.zeros(src_sample_num, 64 * 32, 5, dtype=torch.float32)
y = torch.zeros(src_sample_num, 64 * 32, mode_num, dtype=torch.float32)

xs = torch.linspace(0, 1, 64, device="cuda", dtype=torch.float32)
ys = torch.linspace(0, 1, 32, device="cuda", dtype=torch.float32)
gridx, gridy = torch.meshgrid(xs, ys)
for i in tqdm(range(src_sample_num)):
    while True:
        size_scale = torch.rand(1).cuda()
        freq_scale = torch.rand(1).cuda()
        size_k = size_scale * (size_max - size_min) + size_min
        freq_k = freq_scale * (freq_max - freq_min) + freq_min
        size_k = torch.exp(size_k)
        freq_k = torch.exp(freq_k)
        vertices = vertices_base * size_k
        ks = ks_base * freq_k / size_k
        r_min = 1.5
        r_max = 3.0
        trg_pos = torch.zeros(64, 32, 3, device="cuda", dtype=torch.float32)
        r_scale = torch.rand(1).cuda()
        r = (r_scale * (r_max - r_min) + r_min).item()
        trg_pos[:, :, 0] = r_scale
        trg_pos[:, :, 1] = gridx
        trg_pos[:, :, 2] = gridy
        trg_pos = trg_pos.reshape(-1, 3)

        trg_points = get_spherical_surface_points(vertices, r)
        trg_points = trg_points
        ffat_map, convergence = monte_carlo_process(vertices, ks, trg_points)
        if check_correct:
            ffat_map_bem = bem_process(vertices, ks, trg_points)
        if convergence:
            break

    x[i, :, 0] = size_scale.cpu()
    x[i, :, 1] = freq_scale.cpu()
    x[i, :, 2:] = trg_pos.cpu()
    y[i] = torch.from_numpy(np.abs(ffat_map)).T
    if check_correct:
        for i in range(8):
            v_min, v_max = np.min(np.abs(ffat_map_bem[i])), np.max(
                np.abs(ffat_map_bem[i])
            )
            plt.subplot(2, 8, i + 1)
            plt.imshow(np.abs(ffat_map[i]).reshape(64, 32), vmin=v_min, vmax=v_max)
            plt.colorbar()
            plt.subplot(2, 8, i + 9)
            plt.imshow(np.abs(ffat_map_bem[i]).reshape(64, 32), vmin=v_min, vmax=v_max)
            plt.colorbar()
        plt.savefig(
            f"{data_dir}/compare_{size_scale.item():.2f}_{freq_scale.item():.2f}_{r_scale.item():.2f}.png"
        )
        plt.close()

torch.save({"x": x, "y": y}, f"{data_dir}/data_{sys.argv[1]}.pt")
