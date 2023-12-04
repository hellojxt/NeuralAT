import sys

sys.path.append("./")

import torch
from src.cuda_imp import ImportanceSampler, MonteCarloWeight
from src.loader.model import ModalSoundObject
import numpy as np
from src.visualize import plot_mesh, plot_point_cloud, crop_center, combine_images
from src.solver import BiCGSTAB, BiCGSTAB_batch
import os
import time


OUTPUT_ROOT = f"output/{os.path.basename(__file__)[:-3]}/"
os.makedirs(OUTPUT_ROOT, exist_ok=True)


class Timer:
    def __init__(self, log_output=True):
        self.start_time = time.time()
        self.log_output = log_output

    def get_time(self):
        torch.cuda.synchronize()
        cost_time = time.time() - self.start_time
        self.start_time = time.time()
        return cost_time

    def log(self, *msg):
        if self.log_output:
            print(*msg, " cost time: ", self.get_time())


def run(warm_up=False):
    importance = torch.ones(len(triangles), dtype=torch.float32).cuda()
    timer = Timer(warm_up == False)
    sampler = ImportanceSampler(vertices, triangles, importance, 100000)
    sampler.update()
    sampler.poisson_disk_resample(0.008, 4)
    timer.log("sample points: ", sampler.num_samples)

    G1_batch = torch.empty(
        (batch_size, sampler.num_samples, sampler.num_samples),
        dtype=torch.complex64,
        device="cuda",
    )
    b_batch = torch.empty(
        (batch_size, sampler.num_samples, 1), dtype=torch.complex64, device="cuda"
    )

    mode_idx = 0
    for triangle_neumann, k in zip(triangle_neumanns, ks):
        G0_constructor = MonteCarloWeight(sampler.points, sampler, k)
        G1_constructor = MonteCarloWeight(sampler.points, sampler, k, deriv=True)
        G0 = G0_constructor.get_weights_boundary()
        G1 = G1_constructor.get_weights_boundary()
        G1_batch[mode_idx] = G1
        neumann = sampler.get_points_neumann(triangle_neumann)
        b_batch[mode_idx] = G0 @ neumann
        mode_idx += 1

    timer.log("construct G and b")
    solver = BiCGSTAB_batch(lambda x: (torch.bmm(G1_batch, x) - x))
    timer.log("construct A")

    dirichlet = solver.solve(b_batch, tol=0.001, nsteps=20)
    timer.log("solve")
    if not warm_up:
        image_paths_full = []
        zoom = 2.0
        crop_width = 300
        crop_height = 150
        plot_point_cloud(
            vertices, triangles, sampler.points, dirichlet[0].real, point_size
        ).show()
        for mode_idx in range(batch_size):
            data = dirichlet[mode_idx]
            gt = gts[mode_idx]
            image_paths = []
            img_path = f"{OUTPUT_ROOT}/{mode_idx}_real_gt.png"
            image_paths.append(img_path)
            plot_point_cloud(
                vertices, triangles, vertices, gt.real, point_size, zoom=zoom
            ).write_image(img_path)
            crop_center(img_path, crop_width, crop_height)
            img_path = f"{OUTPUT_ROOT}/{mode_idx}_real.png"
            image_paths.append(img_path)
            plot_point_cloud(
                vertices, triangles, sampler.points, data.real, point_size, zoom=zoom
            ).write_image(img_path)
            crop_center(img_path, crop_width, crop_height)
            img_path = f"{OUTPUT_ROOT}/{mode_idx}_image_gt.png"
            image_paths.append(img_path)
            plot_point_cloud(
                vertices, triangles, vertices, gt.imag, point_size, zoom=zoom
            ).write_image(img_path)
            crop_center(img_path, crop_width, crop_height)
            img_path = f"{OUTPUT_ROOT}/{mode_idx}_image.png"
            image_paths.append(img_path)
            plot_point_cloud(
                vertices, triangles, sampler.points, data.imag, point_size, zoom=zoom
            ).write_image(img_path)
            crop_center(img_path, crop_width, crop_height)

            image_paths_full.append(image_paths)
        combine_images(image_paths_full, f"{OUTPUT_ROOT}/combined.png")


batch_size = 16
obj_id = 2
point_size = 4
sound_object = ModalSoundObject(f"dataset/0000{obj_id}")
vertices = torch.tensor(sound_object.vertices, dtype=torch.float32).cuda()
triangles = torch.tensor(sound_object.triangles, dtype=torch.int32).cuda()
triangle_neumanns = []
ks = []
gts = []
for mode_idx in range(batch_size):
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
    triangle_neumanns.append(triangle_neumann)
    ks.append(k)
    gts.append(gt)

run(warm_up=True)
run(warm_up=False)
