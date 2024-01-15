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
from src.ffat_solve import monte_carlo_solve, bem_solve
from src.audio import calculate_bin_frequencies

data_dir = sys.argv[1]


def get_output_dir(freq, src_pos_y):
    dir_name = f"{data_dir}/{src_pos_y:.1f}_{freq:.0f}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def monte_carlo_process(vertices, ks, trg_points):
    while True:
        timer = Timer()
        ffat_map, convergence = monte_carlo_solve(
            vertices,
            triangles,
            neumann_tri.T,
            ks_batch,
            trg_points,
            8000,
            plot=False,
        )
        cost_time = timer.get_time()
        if convergence:
            break
    np.savez(
        f"{get_output_dir(freq, src_pos_y)}/ours.npz",
        ffat_map=np.abs(ffat_map),
        cost_time=cost_time,
    )


def bem_process(vertices, ks, trg_points):
    timer = Timer()
    ffat_map = bem_solve(
        vertices, triangles, neumann_tri.T, ks_batch, trg_points, plot=False
    )
    cost_time = timer.get_time()
    np.savez(
        f"{get_output_dir(freq, src_pos_y)}/bem.npz",
        vertices=vertices.cpu().numpy(),
        triangles=triangles.cpu().numpy(),
        neumann=neumann_tri.cpu().numpy(),
        wave_number=-ks.cpu().numpy(),
        ffat_map=np.abs(ffat_map),
        cost_time=cost_time,
        points=trg_points.cpu().numpy(),
    )


def calculate_ffat_map_neuPAT(src_pos, trg_pos, freq_pos):
    x = torch.zeros(len(trg_points), 5, dtype=torch.float32, device="cuda")
    x[:, :1] = src_pos[1]
    x[:, 1:4] = trg_pos
    x[:, 4:] = freq_pos
    timer = Timer()
    ffat_map = model(x).T
    cost_time = timer.get_time()
    np.savez(
        f"{get_output_dir(freq, src_pos_y)}/neuPAT.npz",
        ffat_map=ffat_map.cpu().numpy(),
        cost_time=cost_time,
    )


import json
import numpy as np


with open(f"{data_dir}/../config.json", "r") as file:
    config_data = json.load(file)

sample_data = torch.load(f"{data_dir}/../sample_points.pt")
points_static = sample_data["points_static"].cuda()
points_vib = sample_data["points_vib"].cuda()
normal_static = sample_data["normal_static"].cuda()
normal_vib = sample_data["normal_vib"].cuda()
neumann = sample_data["neumann"].cuda()
cdf = sample_data["cdf"].item()
importance = sample_data["importance"].cuda()
vertices_vib = sample_data["vertices_vib"].cuda()
triangles_vib = sample_data["triangles_vib"].cuda()
vertices_static = sample_data["vertices_static"].cuda()
triangles_static = sample_data["triangles_static"].cuda()
neumann_tri_static = torch.zeros(len(triangles_static), 1, dtype=torch.complex64).cuda()
neumann_tri_vib = torch.ones(len(triangles_vib), 1, dtype=torch.complex64).cuda()

triangle_y = vertices_vib[triangles_vib][:, :, 1].mean(1)
neumann_tri_vib[triangle_y > -0.04] = 0

neumann_tri = torch.cat([neumann_tri_vib, neumann_tri_static], dim=0)
triangles = torch.cat([triangles_vib, triangles_static + len(vertices_vib)], dim=0)

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
freq_min = config_data.get("solver", {}).get("freq_min", 100)
freq_max = config_data.get("solver", {}).get("freq_max", 10000)
freq_min_log = np.log10(freq_min)
freq_max_log = np.log10(freq_max)

with open(f"{data_dir}/net.json", "r") as file:
    train_config_data = json.load(file)

train_params = train_config_data.get("train", {})
model = NeuPAT(
    1,
    train_config_data.get("encoding_config"),
    train_config_data.get("network_config"),
).cuda()
model.load_state_dict(torch.load(f"{data_dir}/model.pt"))
model.eval()
torch.set_grad_enabled(False)

first = True
xs = torch.linspace(0, 1, 64, device="cuda", dtype=torch.float32)
ys = torch.linspace(0, 1, 32, device="cuda", dtype=torch.float32)
gridx, gridy = torch.meshgrid(xs, ys)

freq_bins = calculate_bin_frequencies()
print("freq_bins:", freq_bins)
for src_pos_y in range(0, 10):
    src_pos_y = src_pos_y / 10
    print("src_pos_y:", src_pos_y)
    for freq_bin in freq_bins:
        if freq_bin < freq_min or freq_bin > freq_max:
            continue
        freq_pos = (np.log10(freq_bin) - freq_min_log) / (freq_max_log - freq_min_log)
        src_pos = torch.zeros(3, device="cuda", dtype=torch.float32)
        src_pos[1] = src_pos_y
        displacement = src_pos * (src_pos_max - src_pos_min) + src_pos_min
        vertices_vib_updated = vertices_vib + displacement
        vertices = torch.cat([vertices_vib_updated, vertices_static], dim=0)
        r_min = 2
        r_max = 4
        trg_pos = torch.zeros(64, 32, 3, device="cuda", dtype=torch.float32)
        r_scale = torch.ones(1).cuda() * 0.5
        r = (r_scale * (r_max - r_min) + r_min).item()
        trg_pos[:, :, 0] = r_scale
        trg_pos[:, :, 1] = gridx
        trg_pos[:, :, 2] = gridy
        trg_pos = trg_pos.reshape(-1, 3)
        trg_points = get_spherical_surface_points(vertices_static, r)

        freq_log = freq_pos * (freq_max_log - freq_min_log) + freq_min_log
        freq = 10**freq_log
        k = 2 * np.pi * freq / 343.2

        ks_batch = torch.tensor([-k], dtype=torch.float32).cuda()
        if first:
            monte_carlo_process(vertices, ks_batch, trg_points)
            bem_process(vertices, ks_batch, trg_points)
            calculate_ffat_map_neuPAT(src_pos, trg_pos, freq_pos)
            first = False
        monte_carlo_process(vertices, ks_batch, trg_points)
        bem_process(vertices, ks_batch, trg_points)
        calculate_ffat_map_neuPAT(src_pos, trg_pos, freq_pos)
