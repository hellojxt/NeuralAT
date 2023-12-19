import sys

sys.path.append("./")

import torch
from src.cuda_imp import (
    ImportanceSampler,
    MonteCarloWeight,
    FDTDSimulator,
    get_bound_info,
)
from src.kleinpat_loader.model import (
    ModalSoundObject,
    SoundObjList,
    SimpleSoundObject,
    BEMModel,
)
import numpy as np
from src.visualize import plot_mesh, plot_mesh_with_plane
from src.solver import BiCGSTAB
import os
import time


OUTPUT_ROOT = f"output/{os.path.basename(__file__)[:-3]}"
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def FDTD(vertices, triangles, triangle_neumann, omega, res):
    vertices = torch.from_numpy(vertices).cuda().float()
    triangles = torch.from_numpy(triangles).cuda().int()
    triangle_neumann_amp = torch.from_numpy(triangle_neumann).cuda()
    min_bound, max_bound, bound_size = get_bound_info(vertices, padding=1.5)
    fdtd = FDTDSimulator(min_bound, max_bound, bound_size, res)
    step_num = 2000
    triangles_neumann = torch.ones(
        len(triangles), step_num, dtype=torch.float32, device=vertices.device
    )
    start_time = time.time()
    for i in range(step_num):
        triangles_neumann[:, i] = torch.cos(i * fdtd.dt * omega) * triangle_neumann_amp
    fdtd.update(vertices, triangles, triangles_neumann)
    fdtd.accumulate_grids.zero_()

    for i in range(step_num):
        triangles_neumann[:, i] = (
            torch.cos((i + step_num) * fdtd.dt * omega) * triangle_neumann_amp
        )
    fdtd.update(vertices, triangles, triangles_neumann)

    end_time = time.time()
    xs, ys, zs = fdtd.get_mgrid_xyz()
    return (
        fdtd.accumulate_grids.reshape(res, res, res).cpu().numpy(),
        end_time - start_time,
        xs,
        ys,
        zs,
        min_bound,
        max_bound,
    )


def BEM(vertices, triangles, triangle_neumann, omega, target_points, res):
    start_time = time.time()
    bem_model = BEMModel(vertices, triangles, omega / 343.0)
    residual = bem_model.boundary_equation_solve(triangle_neumann)
    dirichlet = bem_model.potential_solve(target_points)
    end_time = time.time()
    return np.abs(dirichlet.reshape(res, res, res)), end_time - start_time


def baseline(vertices, triangles, triangle_neumann, omega, target_points, res):
    vertices = torch.from_numpy(vertices).cuda().float()
    triangles = torch.from_numpy(triangles).cuda().int()
    triangle_neumann = torch.from_numpy(triangle_neumann).cuda().float()
    k = omega / 343.0  # speed of sound
    importance = torch.ones(len(triangles), dtype=torch.float32).cuda()
    sampler = ImportanceSampler(
        vertices, triangles, importance, 100000, triangle_neumann
    )
    start_time = time.time()
    sampler.update()
    sampler.poisson_disk_resample(0.003, 4)
    print("sample points: ", sampler.num_samples)
    G0_constructor = MonteCarloWeight(sampler.points, sampler, k)
    G1_constructor = MonteCarloWeight(sampler.points, sampler, k, deriv=True)
    G0 = G0_constructor.get_weights_boundary()
    G1 = G1_constructor.get_weights_boundary()

    solver = BiCGSTAB(G1)
    neumann_src = sampler.points_neumann.to(torch.complex64).squeeze(-1)
    b = G0 @ neumann_src + neumann_src
    x = torch.zeros_like(b)
    dirichlet_src = solver.solve(b, x, tol=1e-5, nsteps=100)
    end_time = time.time()

    data = []
    step_size = 5000
    idx = 0
    while idx < len(target_points):
        trg_points = target_points[idx : idx + step_size]
        G0_constructor = MonteCarloWeight(trg_points, sampler, k)
        G1_constructor = MonteCarloWeight(trg_points, sampler, k, deriv=True)
        G0 = G0_constructor.get_weights()
        G1 = G1_constructor.get_weights()
        RHS = G1 @ dirichlet_src - G0 @ neumann_src
        data.append(torch.abs(RHS).flatten())
        idx += step_size

    return (
        torch.cat(data).detach().cpu().numpy().reshape(res, res, res) / 2,
        end_time - start_time,
    )


cup = SimpleSoundObject("dataset/cup.obj")
cup.set_neumann(0.0)
sphere = SimpleSoundObject("dataset/sphere.obj")
sphere.set_neumann(1.0)
sphere.scale(0.03)
sphere.translate(0.0, 0.3, 0.0)
scene = SoundObjList([cup, sphere])


freq = 2000
omega = 2 * np.pi * freq
res = 95

# warm up
fdtd_data, fdtd_time, xs, ys, zs, min_bound, max_bound = FDTD(
    scene.vertices, scene.triangles, scene.triangles_neumann, omega, res
)
target_points = torch.stack([xs.flatten(), ys.flatten(), zs.flatten()], dim=1)

baseline_data, baseline_time = baseline(
    scene.vertices, scene.triangles, scene.triangles_neumann, omega, target_points, res
)


fdtd_data, fdtd_time, xs, ys, zs, min_bound, max_bound = FDTD(
    scene.vertices, scene.triangles, scene.triangles_neumann, omega, res
)
target_points = torch.stack([xs.flatten(), ys.flatten(), zs.flatten()], dim=1)

bem_data, bem_time = BEM(
    scene.vertices, scene.triangles, scene.triangles_neumann, omega, target_points, res
)

baseline_data, baseline_time = baseline(
    scene.vertices, scene.triangles, scene.triangles_neumann, omega, target_points, res
)

plot_mesh_with_plane(
    scene.vertices,
    scene.triangles,
    scene.triangles_neumann,
    xs,
    ys,
    zs,
    fdtd_data,
    min_bound,
    max_bound,
    mesh_opacity=0.5,
    cmin=0,
    cmax=0.01,
).show()


plot_mesh_with_plane(
    scene.vertices,
    scene.triangles,
    scene.triangles_neumann,
    xs,
    ys,
    zs,
    bem_data,
    min_bound,
    max_bound,
    mesh_opacity=0.5,
    cmin=0,
    cmax=0.01,
).show()

plot_mesh_with_plane(
    scene.vertices,
    scene.triangles,
    scene.triangles_neumann,
    xs,
    ys,
    zs,
    baseline_data,
    min_bound,
    max_bound,
    mesh_opacity=0.5,
    cmin=0,
    cmax=0.01,
).show()

print("FDTD time: {}".format(fdtd_time))
print("BEM time: {}".format(bem_time))
print("Baseline time: {}".format(baseline_time))
