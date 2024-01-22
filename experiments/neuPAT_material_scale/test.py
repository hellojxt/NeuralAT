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

data_dir = sys.argv[1]


def get_output_dir(size_k, freq_k):
    dir_name = f"{data_dir}/{size_k:.2f}_{freq_k:.2f}"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    return dir_name


def monte_carlo_process(vertices, ks, trg_points):
    timer = Timer()
    ffat_map, _ = monte_carlo_solve(
        vertices, triangles, neumann_tri, ks, trg_points, 3000
    )
    cost_time = timer.get_time()
    print(ffat_map.max(), ffat_map.min())
    np.savez(
        f"{get_output_dir(size_k, freq_k)}/ours.npz",
        ffat_map=np.abs(ffat_map),
        cost_time=cost_time,
    )


def bem_process(vertices, ks, trg_points):
    timer = Timer()
    ffat_map = bem_solve(vertices, triangles, neumann_tri, ks, trg_points)
    cost_time = timer.get_time()
    np.savez(
        f"{get_output_dir(size_k, freq_k)}/bem.npz",
        vertices=vertices.cpu().numpy(),
        triangles=triangles.cpu().numpy(),
        neumann=neumann_tri.cpu().numpy(),
        wave_number=-ks.cpu().numpy(),
        ffat_map=np.abs(ffat_map),
        cost_time=cost_time,
        points=trg_points.cpu().numpy(),
    )


def calculate_ffat_map_neuPAT(size_scale, freq_scale, trg_pos):
    x = torch.zeros(len(trg_points), 5, dtype=torch.float32, device="cuda")
    x[:, 0] = size_scale
    x[:, 1] = freq_scale
    x[:, 2:] = trg_pos
    timer = Timer()
    ffat_map = model(x).T
    print(ffat_map.max(), ffat_map.min())
    cost_time = timer.get_time()
    np.savez(
        f"{get_output_dir(size_k, freq_k)}/neuPAT.npz",
        ffat_map=ffat_map.cpu().numpy(),
        cost_time=cost_time,
    )


import json
import numpy as np


with open(f"{data_dir}/../config.json", "r") as file:
    config_data = json.load(file)

data = torch.load(f"{data_dir}/../modal_data.pt")
vertices_base = data["vertices"]
triangles = data["triangles"]
neumann_tri = data["neumann_tri"]
ks_base = data["ks"]
mode_num = len(ks_base)

size_min = np.log(config_data.get("solver", {}).get("size_scale_min"))
size_max = np.log(config_data.get("solver", {}).get("size_scale_max"))
freq_min = np.log(config_data.get("solver", {}).get("freq_scale_min"))
freq_max = np.log(config_data.get("solver", {}).get("freq_scale_max"))

with open(f"{data_dir}/net.json", "r") as file:
    train_config_data = json.load(file)

train_params = train_config_data.get("train", {})
model = NeuPAT(
    mode_num,
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

for size_scale, freq_scale in [
    (0.5, 0.4),
    (0.5, 0.75),
    (0.5, 1.0),
    (0.7, 1.0),
    (1.0, 1.0),
]:
    size_k = size_scale * (size_max - size_min) + size_min
    freq_k = freq_scale * (freq_max - freq_min) + freq_min
    size_k = np.exp(size_k)
    freq_k = np.exp(freq_k)
    print(size_k, freq_k)
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
    if first:
        monte_carlo_process(vertices, ks, trg_points)
        bem_process(vertices, ks, trg_points)
        calculate_ffat_map_neuPAT(size_scale, freq_scale, trg_pos)
        first = False
    monte_carlo_process(vertices, ks, trg_points)
    bem_process(vertices, ks, trg_points)
    calculate_ffat_map_neuPAT(size_scale, freq_scale, trg_pos)
