import sys

sys.path.append("./")
import torch
from src.cuda_imp import ImportanceSampler, MonteCarloWeight, batch_green_func
import meshio

mesh = meshio.read("data/meshes/cube.obj")
vertices = torch.tensor(mesh.points, dtype=torch.float32).cuda()
triangles = torch.tensor(mesh.cells_dict["triangle"], dtype=torch.int32).cuda()
importance = torch.ones(triangles.shape[0], 1, dtype=torch.float32).cuda()
N = 100
M = 100
k = 1.0
p1_sample = ImportanceSampler(vertices, triangles, importance, N)
p2_sample = ImportanceSampler(vertices, triangles, importance, M)
p1_sample.update()
p2_sample.update()

weights = MonteCarloWeight(p1_sample, p2_sample, k).get_weights()
weights_deriv = MonteCarloWeight(p1_sample, p2_sample, k, deriv=True).get_weights()
green = batch_green_func(
    p1_sample.points, p2_sample.points, p2_sample.points_normals, k
)
green_deriv = batch_green_func(
    p1_sample.points, p2_sample.points, p2_sample.points_normals, k, True
)
EPS = 1e-2


def Green_func(y, x, xn, k):
    r = (x - y).norm()
    if r < EPS:
        return (torch.exp(1j * k * r) / (4 * torch.pi)).real
    return (torch.exp(1j * k * r) / (4 * torch.pi * r)).real


def Green_func_deriv(y, x, xn, k):
    r = (x - y).norm()
    if r < EPS:
        return 0
    ikr = 1j * r * k
    potential = (
        -torch.exp(ikr) / (4 * torch.pi * r * r * r) * (1 - ikr) * (x - y).dot(xn)
    )
    return potential.real


near_points_num = torch.zeros(N, dtype=torch.float32, device=vertices.device)
for i in range(N):
    for j in range(M):
        if (p1_sample.points[i] - p2_sample.points[j]).norm() < EPS:
            near_points_num[i] += 1.0

for i in range(N):
    for j in range(M):
        trg_p = p1_sample.points[i]
        src_p = p2_sample.points[j]
        src_n = p2_sample.points_normals[j]
        weight = Green_func(trg_p, src_p, src_n, k)
        assert torch.abs(weight - green[i, j]) < 1e-5, f"{weight} {green[i, j]}"
        if (p1_sample.points[i] - p2_sample.points[j]).norm() < EPS:
            weight *= 2 * (2 * torch.pi * EPS) / near_points_num[i]
        else:
            weight *= (
                2
                * (
                    p1_sample.cdf[-1]
                    - torch.pi * EPS**2 * p2_sample.points_importance[j]
                )
                / (M - near_points_num[i])
            )
        assert torch.abs(weight - weights[i, j]) < 1e-5, f"{weight} {weights[i, j]}"

        weight_deriv = Green_func_deriv(trg_p, src_p, src_n, k)
        assert (
            torch.abs(weight_deriv - green_deriv[i, j]) < 1e-5
        ), f"{weight_deriv} {green_deriv[i, j]}"
        if (p1_sample.points[i] - p2_sample.points[j]).norm() < EPS:
            weight_deriv *= 2 * (2 * torch.pi * EPS) / near_points_num[i]
        else:
            weight_deriv *= (
                2
                * (
                    p1_sample.cdf[-1]
                    - torch.pi * EPS**2 * p2_sample.points_importance[j]
                )
                / (M - near_points_num[i])
            )
        assert (
            torch.abs(weight_deriv - weights_deriv[i, j]) < 1e-5
        ), f"{weight_deriv} {weights_deriv[i, j]}"
