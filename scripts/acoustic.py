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

sound_object = ModalSoundObject("dataset/00002")
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
points = ffat_cube_points(*sound_object.scaled_bbox(1.25), 128)
ffat_map = bem_model.potential_solve(points)
import matplotlib.pyplot as plt

plt.imshow(ffat_map)
plt.savefig("ffat_map.png")

# model = get_mlps(True)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.9)
# N = 4096
# M = 1024
# max_epochs = 1000

# importance = torch.ones(triangles.shape[0], 1, dtype=torch.float32).cuda()
# sampler_src = ImportanceSampler(vertices, triangles, importance, M, triangle_neumann)
# sampler_trg = ImportanceSampler(vertices, triangles, importance, N)
# G0_constructor = MonteCarloWeight(sampler_trg, sampler_src, k)
# G1_constructor = MonteCarloWeight(sampler_trg, sampler_src, k, deriv=True)

# x0 = torch.zeros(1, 3, dtype=torch.float32).cuda()


# loss_lst = []
# for epoch in range(max_epochs):
#     sampler_src.update()
#     sampler_trg.update()
#     neumann_src = sampler_src.points_neumann
#     # neumann_src = batch_green_func(
#     #     x0, sampler_src.points, sampler_src.points_normals, k, True
#     # ).reshape(-1, 1)

#     src_inputs = torch.cat(
#         [sampler_src.points, sampler_src.points_normals], dim=1
#     ).float()
#     trg_inputs = torch.cat(
#         [sampler_trg.points, sampler_trg.points_normals], dim=1
#     ).float()
#     dirichlet_src = model(src_inputs).float()
#     dirichlet_trg = model(trg_inputs).float()
#     G0 = G0_constructor.get_weights()
#     G1 = G1_constructor.get_weights()
#     LHS = dirichlet_trg - G1 @ dirichlet_src
#     RHS = -G0 @ neumann_src
#     loss = ((LHS - RHS) ** 2).mean()
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     scheduler.step()
#     print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
#     loss_lst.append(loss.item())

# from matplotlib import pyplot as plt

# loss_lst = np.array(loss_lst).reshape(-1, 10).mean(axis=1)
# print(loss_lst.shape)
# plt.plot(loss_lst)
# plt.savefig("loss.png")

# data = torch.cat([LHS, RHS], dim=1).detach().cpu().numpy()
# coords = sampler_trg.points.detach().cpu().numpy()

# get_figure(coords, data).show()
