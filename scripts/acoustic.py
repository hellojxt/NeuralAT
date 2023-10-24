import sys

sys.path.append("./")

import torch
from src.cuda_imp import ImportanceSampler, MonteCarloWeight, batch_green_func
from src.bem.bempp import BEMModel
from src.bem.utils import ffat_cube_points
from src.loader.model import ModalSoundObject
import numpy as np
from src.net import get_mlps
from src.visualize import get_figure

mode_idx = 0

sound_object = ModalSoundObject("dataset/00000")
vertices = torch.tensor(sound_object.vertices, dtype=torch.float32).cuda()
triangles = torch.tensor(sound_object.triangles, dtype=torch.int32).cuda()
triangle_neumann = torch.tensor(
    sound_object.get_triangle_neumann(mode_idx), dtype=torch.float32
).cuda()
k = sound_object.get_wave_number(mode_idx)
print("k: ", k)

bem_model = BEMModel(
    sound_object.vertices,
    sound_object.triangles,
    k,
)
bem_model.boundary_equation_solve(triangle_neumann.cpu().numpy())
dirichlet = bem_model.get_dirichlet_coeff().reshape(-1, 1).real
print(vertices.shape, dirichlet.shape)
get_figure(
    vertices.cpu().numpy(),
    dirichlet.reshape(-1, 1),
).show()

model = get_mlps(True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.9)
N = 2048
M = 1024
max_epochs = 20

importance = torch.ones(triangles.shape[0], 1, dtype=torch.float32).cuda()
sampler_src = ImportanceSampler(vertices, triangles, importance, M, triangle_neumann)
sampler_trg = ImportanceSampler(vertices, triangles, importance, N)
G0_constructor = MonteCarloWeight(sampler_trg, sampler_src, k)
G1_constructor = MonteCarloWeight(sampler_trg, sampler_src, k, deriv=True)

x0 = torch.zeros(1, 3, dtype=torch.float32).cuda()

for epoch in range(max_epochs):
    sampler_src.update()
    sampler_trg.update()
    neumann_src = sampler_src.points_neumann
    # neumann_src = batch_green_func(
    #     x0, sampler_src.points, sampler_src.points_normals, k, True
    # ).reshape(-1, 1)
    dirichlet_src = model(sampler_src.points).float()
    dirichlet_trg = model(sampler_trg.points).float()
    G0 = G0_constructor.get_weights()
    G1 = G1_constructor.get_weights()
    LHS = dirichlet_trg - G1 @ dirichlet_src
    RHS = -G0 @ neumann_src
    loss = ((LHS - RHS) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch: {}, Loss: {}".format(epoch, loss.item()))

data = torch.cat([LHS, RHS], dim=1).detach().cpu().numpy()
coords = sampler_trg.points.detach().cpu().numpy()

get_figure(coords, data).show()
