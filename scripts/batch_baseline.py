import sys

sys.path.append("./")

import torch
from src.cuda_imp import ImportanceSampler, MonteCarloWeight
from src.timer import Timer
import numpy as np
from src.visualize import plot_mesh, plot_point_cloud, crop_center, combine_images
from src.solver import BiCGSTAB, BiCGSTAB_batch, BiCGSTAB_batch2
import os


OUTPUT_ROOT = f"output/{os.path.basename(__file__)[:-3]}/"
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def run(warm_up=False):
    importance = torch.ones(len(triangles), dtype=torch.float32).cuda()
    timer = Timer(warm_up == False)
    sampler = ImportanceSampler(vertices, triangles, importance, 100000)
    sampler.update()
    sampler.poisson_disk_resample(0.004, 4)
    timer.log("sample points: ", sampler.num_samples)

    G0_constructor = MonteCarloWeight(sampler.points, sampler)
    G1_constructor = MonteCarloWeight(sampler.points, sampler, deriv=True)
    G0_batch = G0_constructor.get_weights_boundary_ks(ks)
    G1_batch = G1_constructor.get_weights_boundary_ks(ks)

    neumann = (
        triangle_neumanns[sampler.points_index].to(torch.complex64).T.unsqueeze(-1)
    )
    print(neumann.shape)
    # (batch_size, n, 1)
    b_batch = torch.bmm(G0_batch, neumann).permute(1, 2, 0)
    timer.log("construct G and b", record=True)
    solver = BiCGSTAB_batch(
        lambda x: (torch.bmm(G1_batch, x.permute(2, 0, 1)).permute(1, 2, 0) - x)
    )
    timer.log("construct A", record=True)

    dirichlet = solver.solve(b_batch, tol=1e-6, nsteps=2000)
    timer.log("solve", record=True)
    plot_point_cloud(
        vertices, triangles, sampler.points, neumann[0].real, point_size
    ).show()
    plot_point_cloud(
        vertices, triangles, sampler.points, dirichlet[:, :, 0].real, point_size
    ).show()
    plot_point_cloud(vertices, triangles, vertices, gts[0].real, point_size).show()
    if not warm_up and LOG_IMAGE:
        image_paths_full = []
        zoom = 2.0
        crop_width = 300
        crop_height = 150

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


point_size = 5
data_path = sys.argv[1]
data = np.load(data_path)
vertices = torch.tensor(data["vertices"], dtype=torch.float32).cuda()
triangles = torch.tensor(data["triangles"], dtype=torch.int32).cuda()

batch_size = 16
triangle_neumanns = torch.tensor(data["neumann"], dtype=torch.float32).cuda()
ks = torch.tensor(data["wave_number"], dtype=torch.float32).cuda()
gts = data["dirichlet"].T

LOG_IMAGE = False
run(warm_up=True)
# LOG_IMAGE = True
# run(warm_up=False)
