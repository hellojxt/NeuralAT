import sys

sys.path.append("./")
import commentjson as json
import tinycudann as tcnn
import torch
from src.cuda_imp import UniformSampler, batch_green_func
import meshio

mesh = meshio.read("data/meshes/cube.obj")
vertices = torch.tensor(mesh.points, dtype=torch.float32).cuda()
triangles = torch.tensor(mesh.cells_dict["triangle"], dtype=torch.int32).cuda()
with open("data/config.json") as f:
    config = json.load(f)

model = tcnn.NetworkWithInputEncoding(3, 1, config["encoding"], config["network"])

# model = torch.nn.Sequential(
#     torch.nn.Linear(3, 256),
#     torch.nn.ReLU(),
#     torch.nn.Linear(256, 256),
#     torch.nn.ReLU(),
#     torch.nn.Linear(256, 1),
# ).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
N = 1000
M = 1000
k = 1.0
max_epochs = 500

x0 = torch.zeros(1, 1, 3, dtype=torch.float32).cuda()


# def model(x):
#     x = x.reshape(1, -1, 3)
#     return batch_green_func(x0, x, torch.zeros_like(x), k)


def get_neumann(points, points_normals):
    return batch_green_func(
        x0, points.reshape(1, N * M, 3), points_normals.reshape(1, N * M, 3), k, True
    )


trg_sample = UniformSampler(vertices, triangles, N)
src_sample = UniformSampler(vertices, triangles, N * M)
trg, trg_normal, _ = trg_sample.update()
src, src_normal, inv_pdf = src_sample.update()
import torch.autograd as autograd

with autograd.detect_anomaly():
    for epoch in range(max_epochs):
        trg = trg.reshape(N, 1, 3)
        src = src.reshape(N, M, 3)
        src_normal = src_normal.reshape(N, M, 3)
        inv_pdf = inv_pdf.reshape(N, M)
        LHS = model(trg.reshape(-1, 3)).reshape(N, 1)
        green_func = batch_green_func(trg, src, src_normal, k).reshape(N, M)
        green_func_deriv = batch_green_func(trg, src, src_normal, k, True).reshape(N, M)
        neumann = get_neumann(src, src_normal).reshape(N, M)
        dirichlet = model(src.reshape(-1, 3)).reshape(N, M)
        RHS_Arr = 2 * inv_pdf * (green_func_deriv * dirichlet - green_func * neumann)
        RHS = RHS_Arr.mean(dim=1).reshape(N, 1)
        norm = abs(LHS + RHS).detach() + 1e-4
        loss = ((LHS - RHS) / norm).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
        gt = batch_green_func(
            x0, trg.reshape(1, N, 3), trg_normal.reshape(1, N, 3), k
        ).reshape(N, 1)
        torch.save(trg.reshape(N, 3), "output/trg.pt")
        torch.save(gt, "output/gt.pt")
        torch.save(LHS, "output/LHS.pt")
