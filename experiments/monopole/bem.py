import sys

sys.path.append("./")

import torch
from src.cuda_imp import ImportanceSampler, MonteCarloWeight
from src.timer import Timer
from src.modalsound.model import (
    solve_points_dirichlet,
    MultipoleModel,
    MeshObj,
    BEMModel,
    SNR,
)
import numpy as np
from src.visualize import plot_mesh, plot_point_cloud, CombinedFig
from src.solver import BiCGSTAB, BiCGSTAB_batch, BiCGSTAB_batch2
import os
from glob import glob
from tqdm import tqdm
from numba import njit
import matplotlib.pyplot as plt

data_dir = sys.argv[1]
obj = MeshObj(f"{data_dir}/mesh.obj")
x0 = obj.center
n0 = [0, 1, 0]
freqs = [20, 50, 100, 200, 500, 1000, 2000, 5000]
points = obj.spherical_surface_points(2)
Ms = [0, 1]

ffat_map_bem = np.zeros((len(Ms), len(freqs), len(points)), dtype=np.complex64)
ffat_map_gt = np.zeros((len(Ms), len(freqs), len(points)), dtype=np.complex64)
neumann = np.zeros((len(Ms), len(freqs), len(obj.triangles_center)), dtype=np.complex64)
dirichlet_bem = np.zeros((len(Ms), len(freqs), len(obj.vertices)), dtype=np.complex64)
SNRs = np.zeros((len(Ms), len(freqs)))

mode_num = len(freqs) * len(Ms)
for i, M in enumerate(Ms):
    for j, freq in enumerate(freqs):
        k = freq * 2 * np.pi / 343.2
        model = MultipoleModel(x0, n0, -k, M)
        neumann_coeff = model.solve_neumann(obj.triangles_center, obj.triangles_normal)
        bem = BEMModel(obj.vertices, obj.triangles, -k)
        bem.boundary_equation_solve(neumann_coeff)
        dirichlet_coeff = bem.get_dirichlet_coeff()
        # bem.set_dirichlet(model.solve_dirichlet(obj.vertices).cpu().numpy())
        points_dirichlet = bem.potential_solve(points)
        points_dirichlet_gt = model.solve_dirichlet(points).cpu().numpy()

        ffat_map_bem[i, j] = points_dirichlet
        ffat_map_gt[i, j] = points_dirichlet_gt
        neumann[i, j] = neumann_coeff.cpu().numpy()
        dirichlet_bem[i, j] = dirichlet_coeff
        SNRs[i, j] = SNR(points_dirichlet_gt, points_dirichlet)
        print(f"{i}, {j}, {SNRs[i, j]}dB")


np.save(f"{data_dir}/ffat_map_bem.npy", ffat_map_bem)
np.save(f"{data_dir}/ffat_map_gt.npy", ffat_map_gt)
np.save(f"{data_dir}/neumann.npy", neumann)
np.save(f"{data_dir}/dirichlet_bem.npy", dirichlet_bem)
np.save(f"{data_dir}/SNR_bem.npy", SNRs)
