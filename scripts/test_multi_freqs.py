import sys

sys.path.append("./")
import commentjson as json
import tinycudann as tcnn
import torch
from tqdm import tqdm
from src.assemble import (
    assemble_single_boundary_matrix,
    assemble_double_boundary_matrix,
)
import bempp.api
import meshio
import numpy as np

with open("config/config.json") as f:
    config = json.load(f)


class SinSurfSignal:
    def __init__(self, grid, frequency):
        vertices = torch.from_numpy(grid.vertices.T.astype("float32")).cuda()
        print("vertices:", vertices.shape)
        triangles = torch.from_numpy(grid.elements.T.astype("int32")).cuda()
        print("triangles", triangles.shape)
        phase = vertices[:, 2]
        phase = phase[triangles].mean(axis=1)
        print(phase.shape)
        self.coefficients = torch.sin(phase * frequency)
        self.space = bempp.api.function_space(grid, "DP", 0)

    def plot(self):
        grid_func = bempp.api.GridFunction(
            self.space, coefficients=self.coefficients.cpu().numpy()
        )
        grid_func.plot()


class LearnableDirichlet(torch.nn.Module):
    def __init__(self, n, feature_dim):
        super().__init__()
        self.n = n
        self.feature_dim = feature_dim
        self.value = torch.nn.Parameter(torch.randn([n, feature_dim]))

    def forward(self):
        return self.value


mesh = meshio.read("dataset/test_Dataset/surf_mesh/bowl.sf.obj")
grid = bempp.api.Grid(
    vertices=mesh.points.T,
    elements=mesh.cells_dict["triangle"].T,
)
signal = SinSurfSignal(grid, 20)
# signal.plot()
vertices = torch.from_numpy(grid.vertices.T.astype("float32")).cuda()
triangles = torch.from_numpy(grid.elements.T.astype("int32")).cuda()
feature_dim = 32


def print_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Parameter: {name}, Gradient: {param.grad}")


dirichlet = LearnableDirichlet(triangles.shape[0], feature_dim=feature_dim).cuda()
encoding = tcnn.Encoding(1, config["encoding"])
network = tcnn.Network(encoding.n_output_dims + feature_dim, 1, config["network"])

parameters = (
    list(encoding.parameters())
    + list(network.parameters())
    + list(dirichlet.parameters())
)

optimizer = torch.optim.Adam(parameters, lr=0.001)
neumann = signal.coefficients
max_epoch = 10000

freq_sample_cycle = 1
error_dict = {}
train_epoch_rate = 0.9

for i in tqdm(range(max_epoch)):
    if i % freq_sample_cycle == 0 or (i > max_epoch * train_epoch_rate and i % 10 == 0):
        if i > max_epoch * train_epoch_rate:
            print("loss", loss.item())
            error_dict[wave_number] = loss.item()
        freq = torch.rand([1, 1], device="cuda")
        wave_number = (freq * 40).item()
        single_matrix = assemble_single_boundary_matrix(
            vertices, triangles, wave_number
        )
        double_matrix = assemble_double_boundary_matrix(
            vertices, triangles, wave_number
        )
        A = double_matrix - 0.5 * torch.eye(triangles.shape[0], device="cuda")
        b = (single_matrix @ neumann).unsqueeze(1)
        scale_factor = torch.norm(single_matrix) / torch.norm(A)
        b = b / scale_factor
        # print(scale_factor.item())

    optimizer.zero_grad()
    freq_encode = encoding(freq)
    freq_encode = freq_encode.repeat(triangles.shape[0], 1)
    feats = torch.cat([freq_encode, dirichlet()], axis=1)
    x = network(feats).float()
    residual = A @ x - b
    loss = (residual**2).mean() / (b**2).mean()
    print("loss", loss.item())
    if i < max_epoch * train_epoch_rate:
        loss.backward()
        optimizer.step()


import matplotlib.pyplot as plt

# plot error_dict
plt.plot(list(error_dict.keys()), list(error_dict.values()), "o")
plt.savefig("figure/multi_freqs_error.png")
