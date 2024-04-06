import sys

sys.path.append("./")
from src.mcs.mcs import (
    ImportanceSampler,
    MonteCarloWeight,
    get_weights_boundary_ks_base,
    get_weights_potential_ks_base,
)
from src.utils import plot_point_cloud, plot_mesh, CombinedFig
from src.solver import BiCGSTAB_batch
from src.ffat_solve import bem_solve, monte_carlo_solve
from src.modalsound.model import get_spherical_surface_points, complex_ssim, SNR
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from time import time
import configparser
import meshio
import torch

data_dir = "dataset/NeuPAT/audio"

import json
import numpy as np

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
print("freq_min:", freq_min)
print("freq_max:", freq_max)
trg_sample_num = config_data.get("solver", {}).get("trg_sample_num", 1000)
src_sample_num = config_data.get("solver", {}).get("src_sample_num", 1000)
print("trg_sample_num:", trg_sample_num)
print("src_sample_num:", src_sample_num)

x = torch.zeros(src_sample_num, 64 * 32, 7, dtype=torch.float32)
y = torch.zeros(src_sample_num, 64 * 32, 1, dtype=torch.float32)


def calculate_ffat_map():
    return monte_carlo_solve(
        vertices,
        triangles,
        neumann_tri.T,
        ks_batch,
        trg_points,
        sample_num,
        plot=False,
    )
    # ffat_map = torch.zeros(trg_sample_num, 1, dtype=torch.complex64).cuda()
    # neumann_batch = neumann.unsqueeze(0) * 1e4
    # G0_batch = get_weights_boundary_ks_base(
    #     ks_batch, points, normals, importance, cdf, False
    # )
    # G1_batch = get_weights_boundary_ks_base(
    #     ks_batch, points, normals, importance, cdf, True
    # )
    # b_batch = torch.bmm(G0_batch, neumann_batch).permute(1, 2, 0)
    # solver = BiCGSTAB_batch(
    #     lambda x: (torch.bmm(G1_batch, x.permute(2, 0, 1)).permute(1, 2, 0) - x)
    # )
    # tol = config_data.get("solver", {}).get("tol", 1e-6)
    # nsteps = config_data.get("solver", {}).get("nsteps", 100)
    # dirichlet_batch, convergence = solver.solve(b_batch, tol=tol, nsteps=nsteps)
    # dirichlet_batch = dirichlet_batch.permute(2, 0, 1)
    # if not convergence:
    #     return ffat_map, False
    # G0 = get_weights_potential_ks_base(
    #     ks_batch, trg_points, points, normals, importance, cdf, False
    # )
    # G1 = get_weights_potential_ks_base(
    #     ks_batch, trg_points, points, normals, importance, cdf, True
    # )
    # if check_correct:
    #     CombinedFig().add_points(points, dirichlet_batch.real).show()
    #     CombinedFig().add_points(points, dirichlet_batch.imag).show()
    # ffat_map = G1 @ dirichlet_batch - G0 @ neumann_batch
    # ffat_map = ffat_map.abs().squeeze(-1)

    # return ffat_map * 1e-4, True


def calculate_ffat_map_bem():
    return np.abs(
        bem_solve(vertices, triangles, neumann_tri.T, ks_batch, trg_points, plot=False)
    )


check_correct = True
sample_num = 8000
snrs = []
ssims = []
xs = torch.linspace(0, 1, 64, device="cuda", dtype=torch.float32)
ys = torch.linspace(0, 1, 32, device="cuda", dtype=torch.float32)
gridx, gridy = torch.meshgrid(xs, ys)
for i in tqdm(range(src_sample_num)):
    while True:
        if check_correct:
            src_pos = torch.tensor([0.5, 1.0, 0.5], device="cuda", dtype=torch.float32)
        else:
            src_pos = torch.rand(3, device="cuda", dtype=torch.float32)

        displacement = src_pos * (src_pos_max - src_pos_min) + src_pos_min

        vertices_vib_updated = vertices_vib + displacement
        vertices = torch.cat([vertices_vib_updated, vertices_static], dim=0)
        r_min = 2
        r_max = 4
        if check_correct:
            r_scale = torch.zeros(1).cuda()
        else:
            r_scale = torch.rand(1).cuda()
        trg_pos = torch.zeros(64, 32, 3, device="cuda", dtype=torch.float32)
        r = (r_scale * (r_max - r_min) + r_min).item()
        trg_pos[:, :, 0] = r_scale
        trg_pos[:, :, 1] = gridx
        trg_pos[:, :, 2] = gridy
        trg_pos = trg_pos.reshape(-1, 3)
        trg_points = get_spherical_surface_points(vertices_static, r)

        points_vib_updated = points_vib + displacement
        normal_vib_updated = normal_vib
        points = torch.cat([points_vib_updated, points_static], dim=0)
        normals = torch.cat([normal_vib_updated, normal_static], dim=0)
        if check_correct:
            freq_pos = torch.tensor(
                [i / src_sample_num], device="cuda", dtype=torch.float32
            )
        else:
            freq_pos = torch.rand(1, device="cuda", dtype=torch.float32)
        freq_log = freq_pos * (freq_max_log - freq_min_log) + freq_min_log
        freq = 10**freq_log
        k = 2 * np.pi * freq / 343.2

        ks_batch = torch.tensor([-k], dtype=torch.float32).cuda()
        ffat_map, success = calculate_ffat_map()
        if success:
            ffat_map = np.abs(ffat_map)
            break
    if check_correct:
        CombinedFig().add_mesh(
            vertices, triangles, neumann_tri.abs(), opacity=1.0
        ).add_points(trg_points.reshape(64, 32, 3)[15, 15]).show()
        sys.exit()
        # ffat_map_bem = calculate_ffat_map_bem()
        # snrs.append(SNR(ffat_map_bem.reshape(64, 32), ffat_map.reshape(64, 32)))
        # ssims.append(
        #     complex_ssim(ffat_map_bem.reshape(64, 32), ffat_map.reshape(64, 32))
        # )
        # ffat_map = np.log(ffat_map * 10e6 + 1)
        # ffat_map_bem = np.log(ffat_map_bem * 10e6 + 1)
        # vmax = 3
        # vmin = 1
        # import matplotlib.pyplot as plt

        # plt.subplot(121)
        # plt.imshow(ffat_map_bem.reshape(64, 32))
        # plt.colorbar()
        # plt.subplot(122)
        # plt.imshow(ffat_map.reshape(64, 32))
        # plt.colorbar()
        # plt.title(f"snr: {snrs[-1]:.2f}, ssim: {ssims[-1]:.2f}")
        # os.makedirs(f"{data_dir}/{displacement[1]:.2f}", exist_ok=True)
        # plt.savefig(f"{data_dir}/{displacement[1]:.2f}/{i}.png")
        # plt.close()

    x[i, :, :3] = src_pos.cpu()
    x[i, :, 3:6] = trg_pos.cpu()
    x[i, :, 6:] = freq_pos.cpu()
    y[i] = torch.from_numpy(ffat_map).T

if check_correct:
    print("snr:", np.mean(snrs))
    print("ssim:", np.mean(ssims))
else:
    torch.save({"x": x, "y": y}, f"{data_dir}/data/{sys.argv[1]}.pt")
