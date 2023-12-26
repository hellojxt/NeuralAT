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

data_dir = sys.argv[1]

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


sample_data = torch.load(f"{data_dir}/sample_points.pt")
points_static = sample_data["points_static"].cuda()
points_vib = sample_data["points_vib"].cuda()
normal_static = sample_data["normal_static"].cuda()
normal_vib = sample_data["normal_vib"].cuda()
neumann = sample_data["neumann"].cuda()
cdf = sample_data["cdf"].item()
importance = sample_data["importance"].cuda()
ks = sample_data["ks"].cuda()

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

for i in tqdm(range(src_sample_num)):
    trg_pos = torch.rand(trg_sample_num, 3, device="cuda", dtype=torch.float32)
    trg_points = trg_pos * (trg_pos_max - trg_pos_min) + trg_pos_min
    while True:
        src_rot = sample_uniform_quaternion()
        points_vib_updated = rotate_points(points_vib, src_rot)
        normal_vib_updated = rotate_points(normal_vib, src_rot)
        src_pos = torch.rand(3, device="cuda", dtype=torch.float32)
        displacement = src_pos * (src_pos_max - src_pos_min) + src_pos_min
        points_vib_updated = points_vib_updated + displacement
        if points_vib_updated[:, 1].min() > 0.005:
            trg_points = rotate_points(trg_points, src_rot)
            trg_points = trg_points + displacement
            break

    points = torch.cat([points_vib_updated, points_static], dim=0)
    normals = torch.cat([normal_vib_updated, normal_static], dim=0)

    ffat_map = torch.zeros(mode_num, trg_sample_num, 1, dtype=torch.complex64).cuda()
    idx = 0
    batch_step = 8
    while idx < mode_num:
        ks_batch = ks[idx : idx + batch_step]
        neumann_batch = neumann[idx : idx + batch_step]
        G0_batch = get_weights_boundary_ks_base(
            ks_batch, points, normals, importance, cdf, False
        )
        G1_batch = get_weights_boundary_ks_base(
            ks_batch, points, normals, importance, cdf, True
        )
        b_batch = torch.bmm(G0_batch, neumann_batch).permute(1, 2, 0)
        solver = BiCGSTAB_batch(
            lambda x: (torch.bmm(G1_batch, x.permute(2, 0, 1)).permute(1, 2, 0) - x)
        )
        tol = config_data.get("solver", {}).get("tol", 1e-6)
        nsteps = config_data.get("solver", {}).get("nsteps", 100)
        dirichlet_batch = solver.solve(b_batch, tol=tol, nsteps=nsteps).permute(2, 0, 1)

        G0 = get_weights_potential_ks_base(
            ks_batch, trg_points, points, normals, importance, cdf, False
        )
        G1 = get_weights_potential_ks_base(
            ks_batch, trg_points, points, normals, importance, cdf, True
        )

        ffat_map[idx : idx + batch_step] = G1 @ dirichlet_batch - G0 @ neumann_batch

        idx += batch_step
    ffat_map = ffat_map.abs().squeeze(-1)

    # CombinedFig().add_points(points, neumann[63].real).add_points(
    #     trg_points, ffat_map[63].real
    # ).show()
    x[i, :, :3] = src_pos.cpu()
    x[i, :, 3:6] = trg_pos.cpu()
    x[i, :, 6:10] = src_rot.cpu()
    y[i] = ffat_map.T.cpu()

torch.save({"x": x, "y": y}, f"{data_dir}/data_{sys.argv[2]}.pt")
