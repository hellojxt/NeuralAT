import torch
from .cuda_imp import ImportanceSampler, MonteCarloWeight
from .timer import Timer
from .modalsound.model import (
    solve_points_dirichlet,
    MultipoleModel,
    MeshObj,
    BEMModel,
    SNR,
    complex_ssim,
)
import numpy as np
from .visualize import plot_mesh, plot_point_cloud, CombinedFig
from .solver import BiCGSTAB, BiCGSTAB_batch, BiCGSTAB_batch2
import os
from glob import glob
from tqdm import tqdm
from numba import njit
import matplotlib.pyplot as plt
import configparser
import meshio


def compute_sample_r(vertices, triangles, n):
    vertices = vertices.float()
    triangles = triangles.long()
    edge1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    edge2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    cross_product = torch.cross(edge1, edge2, dim=1)
    triangle_areas = 0.5 * torch.norm(cross_product, dim=1)
    total_area = torch.sum(triangle_areas)
    return (total_area.item() / (2 * n)) ** 0.5


def get_sampler(vertices, triangles, n):
    r = compute_sample_r(vertices, triangles, n)
    importance = torch.ones(len(triangles), dtype=torch.float32).cuda()
    sampler = ImportanceSampler(vertices, triangles, importance, 50000)
    sampler.update()
    sampler.poisson_disk_resample(r, 4)
    return sampler


def monte_carlo_sampler_solve(
    sampler,
    neumann_tri,
    ks,
    trg_points,
    tol=1e-6,
    nsteps=100,
):
    G0_constructor = MonteCarloWeight(sampler.points, sampler)
    G1_constructor = MonteCarloWeight(sampler.points, sampler, deriv=True)
    G0_batch = G0_constructor.get_weights_boundary_ks(ks)
    G1_batch = G1_constructor.get_weights_boundary_ks(ks)
    neumann = neumann_tri[:, sampler.points_index].unsqueeze(-1)
    b_batch = torch.bmm(G0_batch, neumann).permute(1, 2, 0)
    solver = BiCGSTAB_batch(
        lambda x: (torch.bmm(G1_batch, x.permute(2, 0, 1)).permute(1, 2, 0) - x)
    )
    dirichlet, convergence = solver.solve(b_batch, tol=tol, nsteps=nsteps)
    if not convergence:
        return None, False
    dirichlet = dirichlet.permute(2, 0, 1)
    G0_constructor = MonteCarloWeight(trg_points, sampler)
    G1_constructor = MonteCarloWeight(trg_points, sampler, deriv=True)
    G0 = G0_constructor.get_weights_potential_ks(ks)
    G1 = G1_constructor.get_weights_potential_ks(ks)
    ffat_map = G1 @ dirichlet - G0 @ neumann
    return ffat_map.squeeze(-1), convergence


def monte_carlo_solve(
    vertices,
    triangles,
    neumann_tri,
    ks,
    trg_points,
    n,
    tol=1e-6,
    nsteps=500,
):
    sampler = get_sampler(vertices, triangles, n)
    print("sample points: ", sampler.num_samples)
    idx = 0
    batch_step = 8
    mode_num = len(ks)
    ffat_map = torch.zeros(mode_num, len(trg_points), dtype=torch.complex64).cuda()
    neumann_tri = neumann_tri * 1e4
    while idx < mode_num:
        ffat_map_batch, convergence = monte_carlo_sampler_solve(
            sampler,
            neumann_tri[idx : idx + batch_step],
            ks[idx : idx + batch_step],
            trg_points,
            tol=tol,
            nsteps=nsteps,
        )
        if not convergence:
            return None, False
        ffat_map[idx : idx + batch_step] = ffat_map_batch
        idx += batch_step
    return ffat_map.cpu().numpy() * 1e-4, convergence


def bem_solve(
    vertices,
    triangles,
    neumann_tri,
    ks,
    trg_points,
    tol=1e-6,
    nsteps=2000,
):
    vertices = vertices.cpu().numpy()
    triangles = triangles.cpu().numpy()
    neumann_tri = neumann_tri.cpu().numpy() * 1e4
    ffat_map = np.zeros((len(ks), len(trg_points)), dtype=np.complex64)
    for i in range(len(ks)):
        k = ks[i].item()
        neumann_coeff = neumann_tri[i]
        bem = BEMModel(vertices, triangles, k)
        bem.boundary_equation_solve(neumann_coeff, tol=tol, maxiter=nsteps)
        ffat_map[i] = bem.potential_solve(trg_points)
    return ffat_map * 1e-4
