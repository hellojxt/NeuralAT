import sys

sys.path.append("./")
from src.net.model import NeuPAT
import torch
from glob import glob
import torch
from tqdm import tqdm
import json
from src.timer import Timer
import os
from src.modalsound.model import get_spherical_surface_points, BEMModel
from src.cuda_imp import (
    ImportanceSampler,
    MonteCarloWeight,
    get_weights_boundary_ks_base,
    get_weights_potential_ks_base,
)
from src.solver import BiCGSTAB_batch
from src.visualize import plot_point_cloud, plot_mesh, CombinedFig

data_dir = sys.argv[1]


def calculate_ffat_map(size_scale, freq_scale, trg_points):
    ffat_map = torch.zeros(mode_num, len(trg_points), 1, dtype=torch.complex64).cuda()
    idx = 0
    batch_step = 8
    ks = ks_base * np.exp(freq_scale * (freq_max - freq_min) + freq_min)
    size_k = np.exp(size_scale * (size_max - size_min) + size_min)
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
    return ffat_map


def calculate_ffat_map_bem(size_scale, freq_scale, trg_points, ks):
    size_k = np.exp(size_scale * (size_max - size_min) + size_min)
    ffat_map = np.zeros((mode_num, len(trg_points)), dtype=np.complex64)
    vertices = vertices_base * size_k
    trg_points = trg_points * size_k
    ks = ks * np.exp(freq_scale * (freq_max - freq_min) + freq_min)
    for i, idx in enumerate(check_indices):
        k = ks[idx]
        neumann_coeff = neumann_tri[idx]
        bem_model = BEMModel(vertices, triangles, k)
        bem_model.boundary_equation_solve(neumann_coeff)
        ffat_map[i] = bem_model.potential_solve(trg_points)
    return np.abs(ffat_map)


import json
import numpy as np


with open(f"{data_dir}/../config.json", "r") as file:
    config_data = json.load(file)


sample_data = torch.load(f"{data_dir}/../sample_points.pt")
vertices_base = sample_data["vertices"].cpu().numpy()
triangles = sample_data["triangles"].cpu().numpy()
neumann_tri = sample_data["neumann_tri"].cpu().numpy()
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
freq_scale_factor = config_data.get("solver", {}).get("freq_scale_factor")
freq_max = np.log(freq_scale_factor)
freq_min = -freq_max


with open(f"{data_dir}/net.json", "r") as file:
    train_config_data = json.load(file)

train_params = train_config_data.get("train", {})
batch_size = train_params.get("batch_size")

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=data_dir)

model = NeuPAT(
    mode_num,
    train_config_data.get("encoding_config"),
    train_config_data.get("network_config"),
).cuda()

model.load_state_dict(torch.load(f"{data_dir}/model.pt"))
model.eval()
torch.set_grad_enabled(False)

points = get_spherical_surface_points(points_vib, 1.5)

freq_scale = 0.8
size_scale = 0.2
ffat_map_gt = calculate_ffat_map(size_scale, freq_scale, points)
ffat_map_gt = ((ffat_map_gt + 10e-6) / 10e-6).log10()

check_indices = [1, 3, 8]

ffat_map_bem = calculate_ffat_map_bem(
    size_scale, freq_scale, points.cpu().numpy(), ks_base.cpu().numpy()
)
ffat_map_bem = np.log10((ffat_map_bem + 10e-6) / 10e-6)


x = torch.zeros(len(points), 5, dtype=torch.float32, device="cuda")
x[:, 0] = size_scale
x[:, 1] = freq_scale
x[:, 2:] = points

ffat_map_model = model(x).T
timer = Timer(True)
ffat_map_model = model(x).T
timer.log("model")

from matplotlib import pyplot as plt

for i, idx in enumerate(check_indices):
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title("Ground Truth")
    plt.imshow(ffat_map_gt[idx].cpu().numpy().reshape(64, 32))
    plt.subplot(1, 3, 2)
    plt.title("BEM")
    plt.imshow(ffat_map_bem[i].reshape(64, 32))
    plt.subplot(1, 3, 3)
    plt.title("Model")
    plt.imshow(ffat_map_model[idx].detach().cpu().numpy().reshape(64, 32))
    plt.savefig(f"{data_dir}/ffat_map_{freq_scale}_{size_scale}_{idx}.png")
    # CombinedFig().add_points(points, ffat_map_bem[i], cmax=3.0, cmin=1.0).show()
    # CombinedFig().add_points(points, ffat_map_gt[idx], cmax=3.0, cmin=1.0).show()
    # CombinedFig().add_points(points, ffat_map_model[idx], cmax=3.0, cmin=1.0).show()
