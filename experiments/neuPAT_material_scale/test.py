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


def get_output_dir(size_scale, freq_scale):
    dir_name = f"{data_dir}/{size_scale}_{freq_scale}"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    return dir_name


def monte_carlo_process(vertices, ks, trg_points):
    timer = Timer()
    ffat_map, _ = monte_carlo_solve(
        vertices, triangles, neumann_tri, ks, trg_points, 4000
    )
    cost_time = timer.get_time()
    np.savez(
        f"{get_output_dir(size_scale, freq_scale)}/ours.npz",
        ffat_map=np.abs(ffat_map),
        cost_time=cost_time,
    )


def bem_process(vertices, ks, trg_points):
    timer = Timer()
    ffat_map = bem_solve(vertices, triangles, neumann_tri, ks, trg_points)
    cost_time = timer.get_time()
    np.savez(
        f"{get_output_dir(size_scale, freq_scale)}/bem.npz",
        ffat_map=np.abs(ffat_map),
        cost_time=cost_time,
    )


def calculate_ffat_map_neuPAT(size_scale, freq_scale, trg_pos):
    x = torch.zeros(len(trg_points), 5, dtype=torch.float32, device="cuda")
    x[:, 0] = size_scale
    x[:, 1] = freq_scale
    x[:, 2:] = trg_pos
    timer = Timer()
    ffat_map = model(x).T
    cost_time = timer.get_time()
    np.savez(
        f"{get_output_dir(size_scale, freq_scale)}/neuPAT.npz",
        ffat_map=ffat_map.cpu().numpy(),
        cost_time=cost_time,
    )


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

for size_scale in [0.0, 0.5, 1.0]:
    for freq_scale in [0.0, 0.5, 1.0]:
        size_scale = torch.rand(1).cuda()
        freq_scale = torch.rand(1).cuda()
        size_k = size_scale * (size_max - size_min) + size_min
        freq_k = freq_scale * (freq_max - freq_min) + freq_min
        size_k = torch.exp(size_k)
        freq_k = torch.exp(freq_k)
        vertices = vertices_base * size_k
        ks = ks_base * freq_k
        trg_points = get_spherical_surface_points(vertices_base, 1.5)
        trg_pos = (trg_points - trg_pos_min) / (trg_pos_max - trg_pos_min)
        trg_points = trg_points * size_k
        if first:
            monte_carlo_process(vertices, ks, trg_points)
            bem_process(vertices, ks, trg_points)
            calculate_ffat_map_neuPAT(size_scale, freq_scale, trg_pos)
            first = False
            sys.exit(0)
        monte_carlo_process(vertices, ks, trg_points)
        bem_process(vertices, ks, trg_points)
        calculate_ffat_map_neuPAT(size_scale, freq_scale, trg_pos)
