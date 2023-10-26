import sys

sys.path.append("./")

import torch
from src.cuda_imp import ImportanceSampler, MonteCarloWeight, batch_green_func
from src.bem.bempp import BEMModel
from src.bem.utils import ffat_cube_points
from src.loader.model import ModalSoundObject
import numpy as np
from src.net import get_mlps
from src.visualize import plot_mesh

mode_idx = 0

sound_object = ModalSoundObject("dataset/00002")
vertices = torch.tensor(sound_object.vertices, dtype=torch.float32).cuda()
vertiecs_normals = torch.tensor(
    sound_object.vertices_normal, dtype=torch.float32
).cuda()
triangles = torch.tensor(sound_object.triangles, dtype=torch.int32).cuda()
triangle_neumann = torch.tensor(
    sound_object.get_triangle_neumann(mode_idx), dtype=torch.float32
).cuda()
k = sound_object.get_wave_number(mode_idx)
print("k: ", k)

# bem_model = BEMModel(
#     sound_object.vertices,
#     sound_object.triangles,
#     k,
# )
# bem_model.boundary_equation_solve(triangle_neumann.cpu().numpy())
# dirichlet = bem_model.get_dirichlet_coeff().reshape(-1, 1).real
# dirichlet = torch.tensor(dirichlet, dtype=torch.float32).cuda()

dirichlet = vertices[:, 1].reshape(-1, 1)

model = get_mlps(True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.9)
N = 1024
M = 64
max_epochs = 500

importance = torch.abs(triangle_neumann)
sampler_src = ImportanceSampler(vertices, triangles, importance, M, triangle_neumann)
sampler_trg = ImportanceSampler(vertices, triangles, importance, N)
# sampler_trg = sampler_src
G0_constructor = MonteCarloWeight(sampler_trg.points, sampler_src, k)
G1_constructor = MonteCarloWeight(sampler_trg.points, sampler_src, k, deriv=True)

for epoch in range(max_epochs):
    if epoch >= 0:
        sampler_src.update()
        sampler_trg.update()
        G0 = G0_constructor.get_weights()
        G1 = G1_constructor.get_weights()
    neumann_src = sampler_src.points_neumann
    src_inputs = torch.cat(
        [sampler_src.points * 5, sampler_src.points_normals], dim=1
    ).float()
    trg_inputs = torch.cat(
        [sampler_trg.points * 5, sampler_trg.points_normals], dim=1
    ).float()
    dirichlet_src = model(src_inputs).float()
    dirichlet_trg = model(trg_inputs).float()

    LHS = dirichlet_trg
    RHS = G1 @ dirichlet_src - G0 @ neumann_src
    loss = ((LHS - RHS) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    print("Epoch: {}, Loss: {}".format(epoch, loss.item()))


vertices_input = torch.cat([vertices * 5, vertiecs_normals], dim=1).float()
LHS = model(vertices_input)
sampler_src.update()
G0_constructor = MonteCarloWeight(vertices, sampler_src, k)
G1_constructor = MonteCarloWeight(vertices, sampler_src, k, deriv=True)
G0 = G0_constructor.get_weights()
G1 = G1_constructor.get_weights()
neumann_src = sampler_src.points_neumann
src_inputs = torch.cat(
    [sampler_src.points * 5, sampler_src.points_normals], dim=1
).float()
dirichlet_src = model(src_inputs).float()
RHS = G1 @ dirichlet_src - G0 @ neumann_src

data = torch.cat([dirichlet, LHS, RHS], dim=1).detach().cpu().numpy()

plot_mesh(
    vertices.cpu().numpy(), triangles.cpu().numpy(), data=data, cmin=-1, cmax=1
).show()
