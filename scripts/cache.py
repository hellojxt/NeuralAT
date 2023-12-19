import sys

sys.path.append("./")

import torch
from src.cuda_imp import ImportanceSampler, MonteCarloWeight
from src.kleinpat_loader.model import ModalSoundObject
import numpy as np
from src.visualize import plot_mesh, plot_point_cloud
from src.solver import BiCGSTAB
import os
import time

OUTPUT_ROOT = f"output/{os.path.basename(__file__)[:-3]}/"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

point_size = 4


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def get_time(self):
        torch.cuda.synchronize()
        cost_time = time.time() - self.start_time
        self.start_time = time.time()
        return cost_time


def run(warm_up=False):
    triangle_neumann = torch.tensor(
        sound_object.get_triangle_neumann(mode_idx), dtype=torch.float32
    ).cuda()
    k = sound_object.get_wave_number(mode_idx)
    (
        gt_real,
        gt_image,
        gt_error,
        gt_cost_time,
    ) = sound_object.get_vertex_dirichlet(mode_idx)
    gt_real, gt_image = map(torch.tensor, (gt_real, gt_image))
    gt_real = gt_real.cuda()
    gt_image = gt_image.cuda()
    gt = torch.stack([gt_real, gt_image], dim=1).squeeze(-1)
    gt = torch.view_as_complex(gt)
    importance = torch.ones(len(triangles), dtype=torch.float32).cuda()

    timer = Timer()
    sampler = ImportanceSampler(
        vertices, triangles, importance, 100000, triangle_neumann
    )
    sampler.update()
    sampler.poisson_disk_resample(0.005, 4)
    if not warm_up:
        print("sample points: ", sampler.num_samples)
        print("possion disk sampling cost time: ", timer.get_time())

    resample_num = 256
    G0_constructor = MonteCarloWeight(sampler.points, sampler, k)
    G1_constructor = MonteCarloWeight(sampler.points, sampler, k, deriv=True)
    G0_constructor.init_random_states(resample_num)
    G1_constructor.init_random_states(resample_num)
    if not warm_up:
        print("init random states cost time: ", timer.get_time())

    G0 = G0_constructor.get_weights_sparse(resample_num)
    G1 = G1_constructor.get_weights_sparse(resample_num)
    # G0 = G0_constructor.get_weights_boundary()
    # G1 = G1_constructor.get_weights_boundary()
    if not warm_up:
        print("construct G cost time: ", timer.get_time())

    A = lambda x: (torch.matmul(G1, x) - x)
    solver = BiCGSTAB(A)
    if not warm_up:
        print("construct A cost time: ", timer.get_time())

    neumann_src = sampler.points_neumann.to(torch.complex64).squeeze(-1)
    b = G0 @ neumann_src
    if not warm_up:
        print("construct b cost time: ", timer.get_time())

    dirichlet_src = solver.solve(b, tol=0.001, nsteps=100)
    if not warm_up:
        print("solve cost time: ", timer.get_time())
        plot_point_cloud(
            vertices, triangles, sampler.points, dirichlet_src.real, point_size, 1.0
        ).show()
        plot_point_cloud(vertices, triangles, vertices, gt.real, point_size, 1.0).show()


for obj_id in [2]:
    sound_object = ModalSoundObject(f"dataset/0000{obj_id}")
    vertices = torch.tensor(sound_object.vertices, dtype=torch.float32).cuda()
    triangles = torch.tensor(sound_object.triangles, dtype=torch.int32).cuda()
    for mode_idx in range(1):
        # warm up
        run(True)
        # start test
        run()
