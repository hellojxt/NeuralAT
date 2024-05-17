import sys

sys.path.append("./")
from src.net.model import NeuPAT
import torch
from glob import glob
import torch
from tqdm import tqdm
import json
import os
from src.modalobj.model import get_spherical_surface_points
from src.bem.solver import BEM_Solver
from src.utils import Timer

data_dir = "dataset/NeuPAT_new/scale/baseline"


def get_output_dir(size_k, freq_k):
    dir_name = f"{data_dir}/../test/{size_k:.2f}_{freq_k:.2f}"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    return dir_name


def bem_process(vertices, triangles, ks, neumanns, trg_points):
    timer = Timer()
    solver = BEM_Solver(vertices, triangles)
    potentials = []
    wave_numbers = []
    for k, neumann in zip(ks, neumanns):
        dirichlet = solver.neumann2dirichlet(k, neumann)
        potential = solver.boundary2potential(k, neumann, dirichlet, trg_points)
        potentials.append(potential.abs().cpu().numpy())
        wave_numbers.append(-k.item())
    cost_time = timer.get_time_cost()
    np.savez(
        f"{get_output_dir(sizeK_base, freqK_base)}/bem.npz",
        vertices=vertices.cpu().numpy(),
        triangles=triangles.cpu().numpy(),
        neumann=neumanns.cpu().numpy(),
        wave_number=wave_numbers,
        ffat_map=potentials,
        cost_time=cost_time,
        points=trg_points.cpu().numpy(),
    )


def calculate_ffat_map_neuPAT(size_scale, freq_scale, trg_pos):
    x = torch.zeros(len(trg_points), 5, dtype=torch.float32, device="cuda")
    x[:, :3] = trg_pos
    x[:, 3] = size_scale
    x[:, 4] = freq_scale
    timer = Timer()
    ffat_map = model(x).T
    cost_time = timer.get_time_cost()
    np.savez(
        f"{get_output_dir(sizeK_base, freqK_base)}/neuPAT.npz",
        ffat_map=ffat_map.cpu().numpy(),
        cost_time=cost_time,
    )


import json
import numpy as np


with open(f"{data_dir}/../config.json", "r") as file:
    js = json.load(file)
    sample_config = js.get("sample", {})
    obj_config = js.get("vibration_obj", {})
    size_base = obj_config.get("size")

data = torch.load(f"{data_dir}/../modal_data.pt")
vertices_base = data["vertices"]
triangles = data["triangles"]
neumann_vtx = data["neumann_vtx"]
ks_base = data["ks"]
mode_num = len(ks_base)
mode_num = 8

ks_base = ks_base[:mode_num]
neumann_vtx = neumann_vtx[:mode_num]

freq_rate = sample_config.get("freq_rate")
size_rate = sample_config.get("size_rate")
bbox_rate = sample_config.get("bbox_rate")
sample_num = sample_config.get("sample_num")
point_num_per_sample = sample_config.get("point_num_per_sample")


with open(f"{data_dir}/net.json", "r") as file:
    net_config = json.load(file)

model = NeuPAT(mode_num, net_config).cuda()
model.load_state_dict(torch.load(f"{data_dir}/model.pt"))
model.eval()
torch.set_grad_enabled(False)

first = True
xs = torch.linspace(0, 1, 64, device="cuda", dtype=torch.float32)
ys = torch.linspace(0, 1, 32, device="cuda", dtype=torch.float32)
gridx, gridy = torch.meshgrid(xs, ys)

for sizeK_base, freqK_base in [
    (0.5, 0.4),
    (0.5, 0.75),
    (0.5, 1.0),
    (0.7, 1.0),
    (1.0, 1.0),
]:
    freqK = freqK_base * freq_rate
    sizeK = 1.0 / (1 + sizeK_base * (size_rate - 1))
    vertices = vertices_base * sizeK
    ks = ks_base * freqK / sizeK**0.5
    sample_points_base = torch.zeros(64 * 32, 3).cuda()

    sample_points_base[:, 0] = 0.5
    sample_points_base[:, 1] = gridx.reshape(-1)
    sample_points_base[:, 2] = gridy.reshape(-1)

    rs = (sample_points_base[:, 0] * (bbox_rate - 1) + 1) * size_base * 0.7
    theta = sample_points_base[:, 1] * 2 * np.pi - np.pi
    phi = sample_points_base[:, 2] * np.pi
    xs = rs * torch.sin(phi) * torch.cos(theta)
    ys = rs * torch.sin(phi) * torch.sin(theta)
    zs = rs * torch.cos(phi)
    trg_points = torch.stack([xs, ys, zs], dim=-1)

    if first:
        bem_process(vertices, triangles, ks, neumann_vtx, trg_points)
        calculate_ffat_map_neuPAT(sizeK_base, freqK_base, sample_points_base)
        first = False
    bem_process(vertices, triangles, ks, neumann_vtx, trg_points)
    calculate_ffat_map_neuPAT(sizeK_base, freqK_base, sample_points_base)
