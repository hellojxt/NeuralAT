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


def get_output_dir(size_scale, freq_scale):
    dir_name = f"{data_dir}/{size_scale}_{freq_scale}"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    return dir_name


def calculate_ffat_map(size_scale, freq_scale, trg_points):
    ffat_map = torch.zeros(mode_num, len(trg_points), 1, dtype=torch.complex64).cuda()
    idx = 0
    batch_step = 8
    ks = ks_base * np.exp(freq_scale * (freq_max - freq_min) + freq_min)
    print(ks)
    size_k = np.exp(size_scale * (size_max - size_min) + size_min)
    points = points_vib * size_k
    normals = normal_vib
    trg_points = trg_points * size_k
    cdf = cdf_base * size_k**2
    timer = Timer()
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
    cost_time = timer.get_time()
    ffat_map = ffat_map.abs().squeeze(-1)
    fig.add_points(points, neumann[0].real).add_points(trg_points, ffat_map[0].real)
    np.savez(
        f"{get_output_dir(size_scale, freq_scale)}/ours.npz",
        ffat_map=ffat_map.cpu().numpy(),
        cost_time=cost_time,
    )


def calculate_ffat_map_bem(size_scale, freq_scale, trg_points, ks):
    size_k = np.exp(size_scale * (size_max - size_min) + size_min)
    ffat_map = np.zeros((mode_num, len(trg_points)), dtype=np.complex64)
    vertices = vertices_base * size_k
    trg_points = trg_points * size_k
    ks = ks * np.exp(freq_scale * (freq_max - freq_min) + freq_min)
    print(ks)
    timer = Timer()
    for i in range(mode_num):
        k = ks[i]
        neumann_coeff = neumann_tri[i]
        bem_model = BEMModel(vertices, triangles, k)
        bem_model.boundary_equation_solve(neumann_coeff)
        ffat_map[i] = bem_model.potential_solve(trg_points)
    cost_time = timer.get_time()
    fig.add_mesh(vertices, triangles).add_points(trg_points * 1.2, ffat_map[0].real)
    np.savez(
        f"{get_output_dir(size_scale, freq_scale)}/bem.npz",
        ffat_map=ffat_map,
        cost_time=cost_time,
        vertices=vertices,
        triangles=triangles,
        wave_number=ks,
        neumann=neumann_tri,
        points=trg_points,
    )


def calculate_ffat_map_neuPAT(size_scale, freq_scale, trg_points):
    trg_pos = (trg_points - trg_pos_min) / (trg_pos_max - trg_pos_min)
    x = torch.zeros(len(trg_points), 5, dtype=torch.float32, device="cuda")
    x[:, 0] = size_scale
    x[:, 1] = freq_scale
    x[:, 2:] = trg_pos
    timer = Timer(True)
    ffat_map = model(x).T
    cost_time = timer.get_time()
    np.savez(
        f"{get_output_dir(size_scale, freq_scale)}/neuPAT.npz",
        ffat_map=ffat_map.cpu().numpy(),
        cost_time=cost_time,
    )


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
freq_min = np.log(config_data.get("solver", {}).get("freq_scale_min"))
freq_max = np.log(config_data.get("solver", {}).get("freq_scale_max"))

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

first = True
trg_points = get_spherical_surface_points(points_vib, 2)
for size_scale in [0.0, 1.0, 1.3]:
    for freq_scale in [0.5, 0.5, 0.9, 1.3]:
        if first:
            fig = CombinedFig()
            calculate_ffat_map(size_scale, freq_scale, trg_points)
            calculate_ffat_map_bem(
                size_scale, freq_scale, trg_points.cpu().numpy(), ks_base.cpu().numpy()
            )
            calculate_ffat_map_neuPAT(size_scale, freq_scale, trg_points)
            first = False
            fig.show()
            sys.exit(0)
        calculate_ffat_map(size_scale, freq_scale, trg_points)
        calculate_ffat_map_bem(
            size_scale, freq_scale, trg_points.cpu().numpy(), ks_base.cpu().numpy()
        )
        calculate_ffat_map_neuPAT(size_scale, freq_scale, trg_points)
