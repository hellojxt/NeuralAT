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
size_scale_factor = config_data.get("solver", {}).get("size_scale_factor")
size_max = np.log(size_scale_factor)
size_min = -size_max
freq_min = np.log(config_data.get("solver", {}).get("freq_scale_min"))
freq_max = np.log(config_data.get("solver", {}).get("freq_scale_max"))
print("trg_pos_min:", trg_pos_min)
print("trg_pos_max:", trg_pos_max)
trg_sample_num = config_data.get("solver", {}).get("trg_sample_num", 1000)
src_sample_num = config_data.get("solver", {}).get("src_sample_num", 1000)
print("trg_sample_num:", trg_sample_num)
print("src_sample_num:", src_sample_num)


def monte_carlo_process(vertices, ks, trg_points):
    return monte_carlo_solve(vertices, triangles, neumann_tri, ks, trg_points, 4000)


def bem_process(vertices, ks, trg_points):
    return bem_solve(vertices, triangles, neumann_tri, ks, trg_points)


def check_correctness():
    size_rn = torch.rand(1).cuda() * (size_max - size_min) + size_min
    freq_rn = torch.rand(1).cuda() * (freq_max - freq_min) + freq_min
    size_k = torch.exp(size_rn)
    freq_k = torch.exp(freq_rn)
    vertices = vertices_base * size_k
    ks = ks_base * freq_k
    trg_points = get_spherical_surface_points(vertices)
    ffat_map_1, convergence = monte_carlo_process(vertices, ks, trg_points)
    ffat_map_2 = bem_process(vertices, ks, trg_points)
    CombinedFig().add_points(trg_points, ffat_map_1[0].real).show()
    CombinedFig().add_points(trg_points, ffat_map_2[0].real).show()
    CombinedFig().add_points(trg_points, ffat_map_1[0].imag).show()
    CombinedFig().add_points(trg_points, ffat_map_2[0].imag).show()

    plt.subplot(121)
    plt.imshow(np.abs(ffat_map_1[0]).reshape(64, 32))
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(np.abs(ffat_map_2[0]).reshape(64, 32))
    plt.colorbar()
    plt.show()


# check_correctness()
x = torch.zeros(src_sample_num, trg_sample_num, 5, dtype=torch.float32)
y = torch.zeros(src_sample_num, trg_sample_num, mode_num, dtype=torch.float32)
for i in tqdm(range(src_sample_num)):
    while True:
        size_scale = torch.rand(1).cuda()
        freq_scale = torch.rand(1).cuda()
        size_k = size_scale * (size_max - size_min) + size_min
        freq_k = freq_scale * (freq_max - freq_min) + freq_min
        size_k = torch.exp(size_k)
        freq_k = torch.exp(freq_k)
        vertices = vertices_base * size_k
        ks = ks_base * freq_k
        trg_pos = torch.rand(trg_sample_num, 3).cuda()
        trg_points = trg_pos * (trg_pos_max - trg_pos_min) + trg_pos_min
        trg_points = trg_points * size_k
        ffat_map, convergence = monte_carlo_process(vertices, ks, trg_points)
        if convergence:
            break

    x[i, :, 0] = size_scale.cpu()
    x[i, :, 1] = freq_scale.cpu()
    x[i, :, 2:] = trg_pos.cpu()
    y[i] = torch.from_numpy(np.abs(ffat_map)).T

    # ffat_map_2 = bem_process(vertices, ks, trg_points)
    # plt.subplot(121)
    # plt.imshow(np.abs(ffat_map[0]).reshape(50, 20))
    # plt.colorbar()
    # plt.subplot(122)
    # plt.imshow(np.abs(ffat_map_2[0]).reshape(50, 20))
    # plt.colorbar()
    # plt.show()
    # CombinedFig().add_mesh(vertices, triangles).add_points(
    #     trg_points, ffat_map[0].real
    # ).show()

torch.save({"x": x, "y": y}, f"{data_dir}/data_{sys.argv[1]}.pt")
