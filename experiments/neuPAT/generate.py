import sys

sys.path.append("./")

from src.mcs.mcs import (
    ImportanceSampler,
    MonteCarloWeight,
    get_weights_boundary_ks_base,
    get_weights_potential_ks_base,
)
from src.modalsound.model import get_spherical_surface_points
from src.utils import plot_point_cloud, plot_mesh, CombinedFig
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
vertices_bem = data["vertices_bem"].cuda()
triangles_bem = data["triangles_bem"].cuda()
neumann_tri_bem = data["neumann_tri_bem"].cuda()

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

x = torch.zeros(src_sample_num, 64 * 32, 10, dtype=torch.float32)
y = torch.zeros(src_sample_num, 64 * 32, mode_num, dtype=torch.float32)
xs = torch.linspace(0, 1, 64, device="cuda", dtype=torch.float32)
ys = torch.linspace(0, 1, 32, device="cuda", dtype=torch.float32)
gridx, gridy = torch.meshgrid(xs, ys)

check_correct = True


def calculate_ffat_map():
    r_min = 1.5
    r_max = 3.0
    trg_pos = torch.zeros(64, 32, 3, device="cuda", dtype=torch.float32)
    r_scale = torch.rand(1).cuda()
    r = (r_scale * (r_max - r_min) + r_min).item()
    trg_pos[:, :, 0] = r_scale
    trg_pos[:, :, 1] = gridx
    trg_pos[:, :, 2] = gridy
    trg_pos = trg_pos.reshape(-1, 3)
    trg_points = get_spherical_surface_points(vertices_vib, r)
    while True:
        src_rot = sample_uniform_quaternion()
        vertices_vib_updated = rotate_points(vertices_vib, src_rot)
        src_pos = torch.rand(3, device="cuda", dtype=torch.float32)
        displacement = src_pos * (src_pos_max - src_pos_min) + src_pos_min
        vertices_vib_updated = vertices_vib_updated + displacement
        if vertices_vib_updated[:, 1].min() > 0.01:
            trg_points = rotate_points(trg_points, src_rot)
            trg_points = trg_points + displacement
            break
    vertices = torch.cat([vertices_vib_updated, vertices_static], dim=0).cuda()
    triangles = torch.cat(
        [triangles_vib, triangles_static + len(vertices_vib)], dim=0
    ).cuda()

    while True:
        ffat_map, convergence = monte_carlo_solve(
            vertices, triangles, neumann_tri, ks, trg_points, 6000, plot=False
        )
        if convergence:
            break
    ffat_map = torch.from_numpy(np.abs(ffat_map))
    if check_correct:
        vertices = torch.cat([vertices_vib_updated, vertices_bem], dim=0).cuda()
        triangles = torch.cat(
            [triangles_vib, triangles_bem + len(vertices_vib)], dim=0
        ).cuda()
        print(vertices.shape, triangles.shape, neumann_tri_bem.shape, ks.shape)
        ffat_map_bem = np.abs(
            bem_solve(vertices, triangles, neumann_tri_bem, ks, trg_points, plot=False)
        )
        import matplotlib.pyplot as plt

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
        plt.savefig(f"{data_dir}/compare_{idx}.png")
        plt.close()
        CombinedFig().add_mesh(vertices, triangles).add_points(trg_points).show()
    return ffat_map, src_pos, trg_pos, src_rot


for idx in tqdm(range(src_sample_num)):
    ffat_map, src_pos, trg_pos, src_rot = calculate_ffat_map()
    x[idx, :, :3] = src_pos.cpu()
    x[idx, :, 3:6] = trg_pos.cpu()
    x[idx, :, 6:10] = src_rot.cpu()
    y[idx] = ffat_map.T

torch.save({"x": x, "y": y}, f"{data_dir}/data_{sys.argv[1]}.pt")
