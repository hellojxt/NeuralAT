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

data_dir = "dataset/NeuPAT/scale"

import json
import numpy as np


with open(f"{data_dir}/config.json", "r") as file:
    config_data = json.load(file)


sample_data = torch.load(f"{data_dir}/sample_points.pt")
points_vib = sample_data["points_vib"].cuda()
normal_vib = sample_data["normal_vib"].cuda()
neumann = sample_data["neumann"].cuda()
cdf_base = sample_data["cdf"].item()
importance = sample_data["importance"].cuda()
ks_base = sample_data["ks"].cuda() 

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

x = torch.zeros(src_sample_num, trg_sample_num, 5, dtype=torch.float32)
y = torch.zeros(src_sample_num, trg_sample_num, mode_num, dtype=torch.float32)


def generate():
    trg_pos = torch.rand(trg_sample_num, 3, device="cuda", dtype=torch.float32)
    trg_points = trg_pos * (trg_pos_max - trg_pos_min) + trg_pos_min
    ffat_map = torch.zeros(mode_num, trg_sample_num, 1, dtype=torch.complex64).cuda()
    idx = 0
    batch_step = 8
    freq_scale = torch.rand(1, device="cuda", dtype=torch.float32)
    freq_k = torch.exp(freq_scale * (freq_max - freq_min) + freq_min)
    ks = ks_base * freq_k
    size_scale = torch.rand(1, device="cuda", dtype=torch.float32)
    size_k = torch.exp(size_scale * (size_max - size_min) + size_min)
    points = points_vib * size_k
    normals = normal_vib
    trg_points = trg_points * size_k
    cdf = cdf_base * size_k**2
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
        dirichlet_batch, convergence = solver.solve(b_batch, tol=tol, nsteps=nsteps)
        dirichlet_batch = dirichlet_batch.permute(2, 0, 1)
        if not convergence:
            print(freq_k, size_k, ks_batch)
            return None, None, None, None, False
        G0 = get_weights_potential_ks_base(
            ks_batch, trg_points, points, normals, importance, cdf, False
        )
        G1 = get_weights_potential_ks_base(
            ks_batch, trg_points, points, normals, importance, cdf, True
        )

        ffat_map[idx : idx + batch_step] = G1 @ dirichlet_batch - G0 @ neumann_batch

        idx += batch_step
    ffat_map = ffat_map.abs().squeeze(-1)

    # CombinedFig().add_points(points, dirichlet_batch[0].real).add_points(
    #     trg_points, ffat_map[0].real
    # ).show()
    return ffat_map, trg_pos, size_scale, freq_scale, True


for i in tqdm(range(src_sample_num)):
    while True:
        ffat_map, trg_pos, size_scale, freq_scale, success = generate()
        if success:
            break

    x[i, :, 0] = size_scale.cpu()
    x[i, :, 1] = freq_scale.cpu()
    x[i, :, 2:] = trg_pos.cpu()
    y[i] = ffat_map.T.cpu()

torch.save({"x": x, "y": y}, f"{data_dir}/data_{sys.argv[1]}.pt")
