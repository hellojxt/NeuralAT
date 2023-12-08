import sys

sys.path.append("./")

import torch
from src.cuda_imp import (
    ImportanceSampler,
    MonteCarloWeight,
    fast_sparse_matrix_vector_mul,
)
from src.loader.model import ModalSoundObject
from src.timer import Timer
import numpy as np
from src.visualize import plot_mesh, plot_point_cloud, crop_center, combine_images
from src.solver import BiCGSTAB, BiCGSTAB_batch
import os


OUTPUT_ROOT = f"output/{os.path.basename(__file__)[:-3]}/"
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def run(warm_up=False):
    importance = torch.ones(len(triangles), dtype=torch.float32).cuda()
    timer = Timer(warm_up == False)
    sampler = ImportanceSampler(vertices, triangles, importance, 100000)
    sampler.update()
    sampler.poisson_disk_resample(0.005, 4)
    timer.log("sample points: ", sampler.num_samples)

    resample_num = 512
    G0_constructor = MonteCarloWeight(sampler.points, sampler)
    G1_constructor = MonteCarloWeight(sampler.points, sampler, deriv=True)
    G0_constructor.init_random_states(resample_num)
    G1_constructor.init_random_states(resample_num)
    timer.log("init random states")

    col_indices_G0, values_G0 = G0_constructor.get_weights_sparse_ks_fast(
        resample_num, ks
    )
    col_indices_G1, values_G1 = G1_constructor.get_weights_sparse_ks_fast(
        resample_num, ks
    )

    neumann = triangle_neumanns[sampler.points_index].to(torch.complex64)

    b_batch = fast_sparse_matrix_vector_mul(
        col_indices_G0, values_G0, neumann
    ).unsqueeze(1)

    timer.log("construct G and b", record=True)

    solver = BiCGSTAB_batch(
        lambda x: fast_sparse_matrix_vector_mul(
            col_indices_G1, values_G1, x.squeeze(1)
        ).unsqueeze(1)
        - x
    )
    timer.log("construct A", record=True)

    dirichlet = solver.solve(b_batch, tol=0.001, nsteps=40)
    timer.log("solve", record=True)

    if not warm_up and LOG_IMAGE:
        image_paths_full = []
        zoom = 2.0
        crop_width = 300
        crop_height = 150
        plot_point_cloud(
            vertices, triangles, sampler.points, dirichlet[:, :, 0].real, point_size
        ).show()
        for mode_idx in range(batch_size):
            data = dirichlet[:, :, mode_idx]
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
                vertices,
                triangles,
                sampler.points,
                data.real,
                point_size,
                zoom=zoom,
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
                vertices,
                triangles,
                sampler.points,
                data.imag,
                point_size,
                zoom=zoom,
            ).write_image(img_path)
            crop_center(img_path, crop_width, crop_height)
            image_paths_full.append(image_paths)
            print(f"mode {mode_idx} write image done")
        combine_images(image_paths_full, f"{OUTPUT_ROOT}/combined.png")
    return timer.record_time


def generate_data(prefix_idx=0):
    triangle_neumanns = []
    ks = []
    gts = []
    for i in range(batch_size):
        mode_idx = prefix_idx + i
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
    return triangle_neumanns, ks, gts


obj_id = 2
point_size = 4
sound_object = ModalSoundObject(f"dataset/0000{obj_id}")
vertices = torch.tensor(sound_object.vertices, dtype=torch.float32).cuda()
triangles = torch.tensor(sound_object.triangles, dtype=torch.int32).cuda()

batch_size = 32
triangle_neumanns, ks, gts = generate_data()
triangle_neumanns = torch.stack(triangle_neumanns, dim=1)
LOG_IMAGE = False
run(warm_up=True)
LOG_IMAGE = True
time_parallel = run(warm_up=False)
LOG_IMAGE = False

batch_size = 1
time_serial = 0
for i in range(32):
    triangle_neumanns, ks, gts = generate_data(i)
    triangle_neumanns = torch.stack(triangle_neumanns, dim=1)
    run(warm_up=True)
    time_serial += run(warm_up=False)

print(f"parallel: {time_parallel}")
print(f"serial: {time_serial}")
