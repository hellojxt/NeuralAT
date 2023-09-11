import sys

sys.path.append("./")
import commentjson as json
import tinycudann as tcnn
import torch
from src.cuda_imp import ImportanceSampler, MonteCarloWeight, batch_green_func
import meshio

mesh = meshio.read("data/meshes/cube.obj")
vertices = torch.tensor(mesh.points, dtype=torch.float32).cuda()
triangles = torch.tensor(mesh.cells_dict["triangle"], dtype=torch.int32).cuda()
importance = torch.ones(triangles.shape[0], 1, dtype=torch.float32).cuda()
with open("data/config.json") as f:
    config = json.load(f)


# model = tcnn.NetworkWithInputEncoding(3, 1, config["encoding"], config["network"])

encoding = tcnn.Encoding(3, config["encoding"], dtype=torch.float32)
network = torch.nn.Sequential(
    torch.nn.Linear(encoding.n_output_dims, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1),
)
model = torch.nn.Sequential(encoding, network).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
N = 2000
M = 5000
k = 30
max_epochs = 300

x0 = torch.zeros(1, 3, dtype=torch.float32).cuda()


# def model(x):
#     x = x.reshape(1, -1, 3)
#     return batch_green_func(x0, x, torch.zeros_like(x), k)


def get_neumann(points, points_normals):
    return batch_green_func(x0, points, points_normals, k, True).reshape(-1, 1)


sampler = ImportanceSampler(vertices, triangles, importance, M)
G0_constructor = MonteCarloWeight(sampler, sampler, k, N, M)
G1_constructor = MonteCarloWeight(sampler, sampler, k, N, M, True)

import torch.autograd as autograd


# with autograd.detect_anomaly():
for epoch in range(max_epochs):
    sampler.update()
    neumann = get_neumann(sampler.points, sampler.points_normals)
    dirichlet = model(sampler.points).float()
    G0 = G0_constructor.get_weights()
    G1 = G1_constructor.get_weights()
    LHS = dirichlet[:N]
    RHS = G1 @ dirichlet - G0 @ neumann
    # norm = abs(LHS + RHS).detach() + 1e-4
    loss = ((LHS - RHS)).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
    gt = batch_green_func(
        x0, sampler.points[:N], sampler.points_normals[:N], k
    ).reshape(N, 1)

    torch.save(sampler.points[:N], "output/points.pt")
    torch.save(gt, "output/gt.pt")
    torch.save(LHS, "output/LHS.pt")
    torch.save(RHS, "output/RHS.pt")
