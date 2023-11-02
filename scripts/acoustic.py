import sys

sys.path.append("./")

import torch
from src.cuda_imp import Sampler
from src.bem.bempp import BEMModel
from src.bem.utils import ffat_cube_points
from src.loader.model import ModalSoundObject
import numpy as np
from src.net import get_mlps
from src.visualize import plot_mesh, plot_point_cloud
import os

mode_idx = 4
sound_object = ModalSoundObject("dataset/00002")
vertices = torch.tensor(sound_object.vertices, dtype=torch.float32).cuda()
triangles = torch.tensor(sound_object.triangles, dtype=torch.int32).cuda()
triangle_neumann = torch.tensor(
    sound_object.get_triangle_neumann(mode_idx), dtype=torch.float32
).cuda()
k = sound_object.get_wave_number(mode_idx)
print("k: ", k)


dirichlet_path = os.path.join(sound_object.dirichlet_dir, f"{mode_idx}.npy")
if not os.path.exists(dirichlet_path):
    bem_model = BEMModel(
        sound_object.vertices,
        sound_object.triangles,
        k,
    )
    bem_model.boundary_equation_solve(triangle_neumann.cpu().numpy())
    dirichlet = bem_model.get_dirichlet_coeff().reshape(-1, 1).real
    np.save(dirichlet_path, dirichlet)
dirichlet_gt = np.load(dirichlet_path)

model = get_mlps(True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.8)
max_epochs = 100

importance = torch.ones(triangles.shape[0], 1, dtype=torch.float32).cuda()
points_sampler = Sampler(vertices, triangles, triangle_neumann, importance, 2)


import time

torch.autograd.set_detect_anomaly(True)
for epoch in range(max_epochs):
    points_sampler.sample_points(k, candidate_num=1, reuse_num=0)
    dirichlet_trg = model(points_sampler.trg_points)
    dirichlet_src = model(points_sampler.src_points)

    # print(dirichlet_trg.max(), dirichlet_trg.min())
    # print(dirichlet_src.max(), dirichlet_src.min())
    # print(points_sampler.A.max(), points_sampler.A.min())
    # print(points_sampler.B.max(), points_sampler.B.min())
    LHS = dirichlet_trg
    RHS = points_sampler.A * dirichlet_src + points_sampler.B
    loss = ((0.5 * LHS - RHS) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    print("epoch: {}, loss: {}".format(epoch, loss.item()))

vertices_normal = torch.tensor(sound_object.vertices_normal).cuda()
dirichlet_vertices = model(torch.cat([vertices, vertices_normal], dim=1))
dirichlet = dirichlet_vertices.detach().cpu().numpy()
features = np.concatenate(
    [dirichlet_gt.reshape(-1, 1), dirichlet.reshape(-1, 1)], axis=1
)
plot_mesh(
    vertices.cpu().numpy(),
    triangles.cpu().numpy(),
    features,
).show()
