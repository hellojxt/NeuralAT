import sys

sys.path.append("./")

import torch
from src.cuda_imp import ImportanceSampler, MonteCarloWeight
from src.loader.model import ModalSoundObject
import numpy as np
from src.net import get_mlps
from src.visualize import plot_mesh
import os
import time

OUTPUT_ROOT = f"output/{os.path.basename(__file__)[:-3]}/"
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def train():
    model = get_mlps(2, True)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, 0.9)
    importance = torch.abs(triangle_neumann)
    sampler_src = ImportanceSampler(
        vertices, triangles, importance, M, triangle_neumann
    )
    sampler_trg = ImportanceSampler(vertices, triangles, importance, N)
    G0_constructor = MonteCarloWeight(sampler_trg.points, sampler_src, k)
    G1_constructor = MonteCarloWeight(sampler_trg.points, sampler_src, k, deriv=True)
    start_time = time.time()
    for epoch in range(max_epochs):
        if epoch % 500 == 0:
            sampler_src.update()
            sampler_trg.update()
            G0 = G0_constructor.get_weights()
            G1 = G1_constructor.get_weights()
        neumann_src = sampler_src.points_neumann.to(torch.complex64).squeeze(-1)
        src_inputs = torch.cat(
            [sampler_src.points * 5, sampler_src.points_normals], dim=1
        ).float()
        trg_inputs = torch.cat(
            [sampler_trg.points * 5, sampler_trg.points_normals], dim=1
        ).float()
        dirichlet_src = torch.view_as_complex(model(src_inputs).float())
        dirichlet_trg = torch.view_as_complex(model(trg_inputs).float())
        # print(
        #     G0.shape,
        #     G1.shape,
        #     dirichlet_src.shape,
        #     dirichlet_trg.shape,
        #     neumann_src.shape,
        # )

        LHS = dirichlet_trg
        # RHS = G1 @ dirichlet_src - G0 @ neumann_src
        # loss = ((LHS - RHS).abs()).mean()
        loss = ((LHS).abs()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print("Epoch: {}, Loss: {}".format(epoch, loss.item()))

    end_time = time.time()
    vertices_input = torch.cat([vertices * 5, vertices_normals], dim=1).float()
    LHS = torch.view_as_complex(model(vertices_input))
    sampler_src.update()
    G0_constructor = MonteCarloWeight(vertices, sampler_src, k)
    G1_constructor = MonteCarloWeight(vertices, sampler_src, k, deriv=True)
    G0 = G0_constructor.get_weights()
    G1 = G1_constructor.get_weights()
    neumann_src = sampler_src.points_neumann.to(torch.complex64).squeeze(-1)
    src_inputs = torch.cat(
        [sampler_src.points * 5, sampler_src.points_normals], dim=1
    ).float()
    dirichlet_src = torch.view_as_complex(model(src_inputs).float())
    RHS = G1 @ dirichlet_src - G0 @ neumann_src
    data = (
        torch.stack(
            [
                gt.real,
                LHS.real,
                RHS.real,
                gt.imag,
                LHS.imag,
                RHS.imag,
            ],
            dim=1,
        )
        .detach()
        .cpu()
        .numpy()
    )
    fig = plot_mesh(
        vertices.cpu().numpy(),
        triangles.cpu().numpy(),
        data=data,
        names=[
            "Ground Truth real",
            "LHS real",
            "RHS real",
            "Ground Truth image",
            "LHS image",
            "RHS image",
        ],
    )
    output_dir = OUTPUT_ROOT + str(obj_id)
    os.makedirs(output_dir, exist_ok=True)
    fig.write_image(f"{output_dir}/mode_{mode_idx}.png", width=1920, height=1080)
    fig.write_html(f"{output_dir}/mode_{mode_idx}.html")
    RHS_rerr = torch.abs(RHS - gt).mean() / torch.abs(gt).mean()
    LHS_rerr = torch.abs(LHS - gt).mean() / torch.abs(gt).mean()
    return RHS_rerr.item(), LHS_rerr.item(), end_time - start_time


N = 10240
M = 10240
max_epochs = 10000

RHS_rerr_lst = []
LHS_rerr_lst = []
time_cost_lst = []
for obj_id in [0, 1, 2, 3]:
    sound_object = ModalSoundObject(f"dataset/0000{obj_id}")
    vertices = torch.tensor(sound_object.vertices, dtype=torch.float32).cuda()
    triangles = torch.tensor(sound_object.triangles, dtype=torch.int32).cuda()
    vertices_normals = torch.tensor(
        sound_object.vertices_normal, dtype=torch.float32
    ).cuda()
    RHS_mode_rerr_lst = []
    LHS_mode_rerr_lst = []
    time_cost_mode_lst = []
    for mode_idx in range(5):
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
        # clear cache
        torch.cuda.empty_cache()
        RHS_rerr, LHS_rerr, cost_time = train()
        RHS_mode_rerr_lst.append(RHS_rerr)
        LHS_mode_rerr_lst.append(LHS_rerr)
        time_cost_mode_lst.append(cost_time)
    RHS_rerr_lst.append(RHS_mode_rerr_lst)
    LHS_rerr_lst.append(LHS_mode_rerr_lst)
    time_cost_lst.append(time_cost_mode_lst)

import matplotlib.pyplot as plt
import numpy as np


def plot_list(output_path, rerr_lst):
    # Create a single figure for all objects
    fig, ax = plt.subplots(figsize=(10, 5))

    # Iterate over obj_id
    for obj_id, mode_rerr_lst_obj in enumerate(rerr_lst):
        # Create an array to store the mode indices (x-axis)
        mode_indices = np.arange(1, len(mode_rerr_lst_obj) + 1)
        rerr_values = [rerr for rerr in mode_rerr_lst_obj]

        # Calculate the x-axis positions for the bars
        x_positions = (
            mode_indices + obj_id * 0.2
        )  # Adjust the 1.2 for spacing between bars

        # Use the 'bar' function to create a bar chart
        ax.bar(x_positions, rerr_values, width=0.2, label=f"Object {obj_id + 1}")

    # Set x-axis labels
    ax.set_xticks(
        [
            (i + 1) + (len(mode_rerr_lst_obj) * 0.1) / 2
            for i in range(len(mode_rerr_lst_obj))
        ]
    )
    ax.set_xticklabels([str(i) for i in range(1, len(mode_rerr_lst_obj) + 1)])
    ax.set_xlabel("Mode Index")
    ax.set_ylabel("Value")
    # y-axis log scale
    # ax.set_yscale("log")
    # Add a legend to distinguish between objects
    ax.legend()

    plt.savefig(output_path, dpi=300)
    plt.close()


plot_list(f"{OUTPUT_ROOT}/RHS_rerr.png", RHS_rerr_lst)
plot_list(f"{OUTPUT_ROOT}/LHS_rerr.png", LHS_rerr_lst)
plot_list(f"{OUTPUT_ROOT}/time_cost.png", time_cost_lst)
np.savetxt(f"{OUTPUT_ROOT}/RHS_rerr.npy", np.array(RHS_rerr_lst))
np.savetxt(f"{OUTPUT_ROOT}/LHS_rerr.npy", np.array(LHS_rerr_lst))
np.savetxt(f"{OUTPUT_ROOT}/time_cost.npy", np.array(time_cost_lst))
