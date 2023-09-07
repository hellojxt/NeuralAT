import sys

sys.path.append("./")
import torch
from src.cuda_imp import UniformSampler, batch_green_func
import meshio

mesh = meshio.read("data/meshes/cube.obj")
vertices = torch.tensor(mesh.points, dtype=torch.float32).cuda()
triangles = torch.tensor(mesh.cells_dict["triangle"], dtype=torch.int32).cuda()
N = 100
M = 32
p1_sample = UniformSampler(vertices, triangles, N)
p2_sample = UniformSampler(vertices, triangles, N * M)
p1, _, _ = p1_sample.update()
p2, p2_normal, _ = p2_sample.update()
p1 = p1.reshape(N, 1, 3)
p2 = p2.reshape(N, M, 3)
p2_normal = p2_normal.reshape(N, M, 3)

green_values = batch_green_func(p1, p2, p2_normal, 1.0)
green_deriv_values = batch_green_func(p1, p2, p2_normal, 1.0, deriv=True)


def Green_func(y, x, xn, k):
    r = (x - y).norm()
    if r < 1e-6:
        return 0
    return (torch.exp(1j * k * r) / (4 * torch.pi * r)).real


def Green_func_deriv(y, x, xn, k):
    r = (x - y).norm()
    if r < 1e-6:
        return 0
    ikr = 1j * r * k
    potential = (
        -torch.exp(ikr) / (4 * torch.pi * r * r * r) * (1 - ikr) * (x - y).dot(xn)
    )
    return potential.real


for i in range(N):
    for j in range(M):
        gt_deriv = Green_func_deriv(p1[i, 0], p2[i, j], p2_normal[i, j], 1.0)
        cuda_deriv = green_deriv_values[i, 0, j]
        assert abs(gt_deriv - cuda_deriv) < 1e-6
        gt = Green_func(p1[i, 0], p2[i, j], p2_normal[i, j], 1.0)
        cuda = green_values[i, 0, j]
        assert abs(gt - cuda) < 1e-6
