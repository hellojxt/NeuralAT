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


mesh = meshio.read("dataset/surf_mesh/bowl.sf.obj")
grid = bempp.api.Grid(mesh.points.T, mesh.cells_dict["triangle"].T)
signal = SinSurfSignal(grid, 20)
# signal.plot()
vertices = torch.from_numpy(grid.vertices.T.astype("float32")).cuda()
triangles = torch.from_numpy(grid.elements.T.astype("int32")).cuda()
feature_dim = 32


def print_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Parameter: {name}, Gradient: {param.grad}")


dirichlet = LearnableDirichlet(triangles.shape[0], feature_dim).cuda()
encoding = tcnn.Encoding(1, config["encoding"])
network = tcnn.Network(encoding.n_output_dims + feature_dim, 1, config["network"])

parameters = (
    list(encoding.parameters())
    + list(network.parameters())
    + list(dirichlet.parameters())
)
optimizer = torch.optim.Adam(parameters, lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

# batch_size = 64
neumann = signal.coefficients
x = torch.rand([1, 1], device="cuda")
wave_number = (x * 20).item()
single_matrix = assemble_single_boundary_matrix(vertices, triangles, wave_number)
double_matrix = assemble_double_boundary_matrix(vertices, triangles, wave_number)


# get normalize factor
with torch.no_grad():
    freq_encode = encoding(x)
    freq_encode = freq_encode.repeat(triangles.shape[0], 1)
    feats = torch.cat([freq_encode, dirichlet()], axis=1)
    print("feats", feats.shape)
    dirichlet_pred = network(feats)
    Ax = -0.5 * dirichlet_pred + double_matrix * dirichlet_pred
    b = single_matrix * neumann
    scale_factor = torch.norm(b) / torch.norm(Ax)
    print("norm of double matrix", torch.norm(double_matrix))
    print("norm of b", torch.norm(b))
    print("norm of Ax", torch.norm(Ax))
    print("scale factor", scale_factor)

print(double_matrix[:10, :10])

max_epoch = 2000

for i in tqdm(range(max_epoch)):
    optimizer.zero_grad()
    freq_encode = encoding(x)
    freq_encode = freq_encode.repeat(triangles.shape[0], 1)
    feats = torch.cat([freq_encode, dirichlet()], axis=1)
    dirichlet_pred = network(feats)
    Ax = -0.5 * dirichlet_pred + double_matrix * dirichlet_pred
    b = single_matrix * neumann / scale_factor
    residual = (Ax - b) ** 2
    loss = residual.mean() / (b**2).mean()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    if i % 100 == 0:
        print("loss", loss.item())
        # print_grad(network)
        # print_grad(encoding)
        # print_grad(dirichlet)

# import matplotlib.pyplot as plt

# x = torch.linspace(0, 0.2, 1000, device="cuda").reshape(-1, 1)
# y = sig(x).float()
# y_pred = model(x).float()
# print(x.shape, y.shape, y_pred.shape)
# plt.plot(x.cpu().numpy(), y.cpu().numpy(), label="True")
# plt.plot(x.cpu().numpy(), y_pred.cpu().detach().numpy(), label="Predicted")

# plt.legend()
# plt.show()
