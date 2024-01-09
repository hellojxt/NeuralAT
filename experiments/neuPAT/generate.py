import sys

sys.path.append("./")

from src.cuda_imp import (
    ImportanceSampler,
    MonteCarloWeight,
    get_weights_boundary_ks_base,
    get_weights_potential_ks_base,
)
from src.modalsound.model import get_spherical_surface_points
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
from src.ffat_solve import monte_carlo_solve, bem_solve

data_dir = "dataset/NeuPAT/bowl"

import json
import numpy as np


def sample_uniform_quaternion():
    """
    Sample a single uniformly distributed quaternion.
    """
    # Uniformly distributed values in [0, 1]
    u1, u2, u3 = torch.rand(3, device="cuda")

    # Spherical to Cartesian coordinates conversion for uniform 3D distribution
    sqrt1_u1 = torch.sqrt(1 - u1)
    sqrt_u1 = torch.sqrt(u1)
    theta1 = 2 * torch.pi * u2
    phi = torch.acos(2 * u3 - 1)

    # Cartesian coordinates on a 3D sphere
    x = sqrt1_u1 * torch.sin(theta1)
    y = sqrt1_u1 * torch.cos(theta1)
    z = sqrt_u1 * torch.sin(phi)
    w = sqrt_u1 * torch.cos(phi)

    return torch.tensor([w, x, y, z], device="cuda")


def rotate_points(points, quaternion):
    """
    Rotate points by a given quaternion.
    Points and quaternion should be on CUDA.
    """
    # Normalize the quaternion
    q_norm = quaternion / quaternion.norm()
    q_vec = q_norm[1:]
    q_scalar = q_norm[0]

    # Extend q_vec to match the dimensions of points
    q_vec = q_vec.view(1, 3).expand(points.size(0), 3)

    # Quaternion multiplication (q * p * q^*)
    q_vec_cross_p = torch.cross(q_vec, points, dim=-1)
    return (
        points
        + 2 * q_scalar * q_vec_cross_p
        + 2 * torch.cross(q_vec, q_vec_cross_p, dim=-1)
    )


with open(f"{data_dir}/config.json", "r") as file:
    config_data = json.load(file)

data = torch.load(f"{data_dir}/modal_data.pt")
vertices_vib = data["vertices_vib"].cuda()
triangles_vib = data["triangles_vib"].cuda()
vertices_static = data["vertices_static"].cuda()
triangles_static = data["triangles_static"].cuda()
neumann_tri = data["neumann_tri"].cuda()
ks = data["ks"].cuda()
mode_num = len(ks)
src_pos_min = torch.tensor(
    config_data.get("solver", {}).get("src_pos_min"), device="cuda", dtype=torch.float32
)
src_pos_max = torch.tensor(
    config_data.get("solver", {}).get("src_pos_max"), device="cuda", dtype=torch.float32
)
trg_pos_min = torch.tensor(
    config_data.get("solver", {}).get("trg_pos_min"), device="cuda", dtype=torch.float32
)
trg_pos_max = torch.tensor(
    config_data.get("solver", {}).get("trg_pos_max"), device="cuda", dtype=torch.float32
)

print("src_pos_min:", src_pos_min)
print("src_pos_max:", src_pos_max)
print("trg_pos_min:", trg_pos_min)
print("trg_pos_max:", trg_pos_max)
trg_sample_num = config_data.get("solver", {}).get("trg_sample_num", 1000)
src_sample_num = config_data.get("solver", {}).get("src_sample_num", 1000)
print("trg_sample_num:", trg_sample_num)
print("src_sample_num:", src_sample_num)

x = torch.zeros(src_sample_num, trg_sample_num, 10, dtype=torch.float32)
y = torch.zeros(src_sample_num, trg_sample_num, mode_num, dtype=torch.float32)


def calculate_ffat_map():
    trg_pos = torch.rand(trg_sample_num, 3, device="cuda", dtype=torch.float32)
    trg_points = trg_pos * (trg_pos_max - trg_pos_min) + trg_pos_min
    while True:
        src_rot = sample_uniform_quaternion()
        vertices_vib_updated = rotate_points(vertices_vib, src_rot)
        src_pos = torch.rand(3, device="cuda", dtype=torch.float32)
        displacement = src_pos * (src_pos_max - src_pos_min) + src_pos_min
        vertices_vib_updated = vertices_vib_updated + displacement
        if vertices_vib_updated[:, 1].min() > 0.001:
            trg_points = rotate_points(trg_points, src_rot)
            trg_points = trg_points + displacement
            break
    vertices = torch.cat([vertices_vib_updated, vertices_static], dim=0).cuda()
    triangles = torch.cat(
        [triangles_vib, triangles_static + len(vertices_vib)], dim=0
    ).cuda()

    while True:
        try:
            ffat_map, convergence = monte_carlo_solve(
                vertices, triangles, neumann_tri, ks, trg_points, 4000
            )
        except RuntimeError:
            continue
        if convergence:
            break
    ffat_map = torch.from_numpy(np.abs(ffat_map))
    # ffat_map_bem = np.abs(bem_solve(vertices, triangles, neumann_tri, ks, trg_points))
    # import matplotlib.pyplot as plt
    # plt.subplot(121)
    # plt.imshow(ffat_map[0].reshape(50, 20))
    # plt.colorbar()
    # plt.subplot(122)
    # plt.imshow(ffat_map_bem[0].reshape(50, 20))
    # plt.colorbar()
    # plt.show()
    return ffat_map, src_pos, trg_pos, src_rot


for i in tqdm(range(src_sample_num)):
    ffat_map, src_pos, trg_pos, src_rot = calculate_ffat_map()
    x[i, :, :3] = src_pos.cpu()
    x[i, :, 3:6] = trg_pos.cpu()
    x[i, :, 6:10] = src_rot.cpu()
    y[i] = ffat_map.T

torch.save({"x": x, "y": y}, f"{data_dir}/data_{sys.argv[1]}.pt")
