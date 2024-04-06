import sys

sys.path.append("./")
from src.bem.solver import BEM_Solver, get_potential_of_sources, solve_linear_equation
import torch
import numpy as np
import meshio
from src.utils import (
    Visualizer,
    normalize_mesh,
    get_triangle_centers,
    get_triangle_normals,
)

mesh_path = "dataset/meshes/sphere.obj"
mesh = meshio.read(mesh_path)
vertices = torch.from_numpy(mesh.points).cuda().float()
vertices = normalize_mesh(vertices)
triangles = torch.from_numpy(mesh.cells_dict["triangle"]).cuda().int()
print(vertices.shape, triangles.shape)
cuda_bem = BEM_Solver(vertices, triangles)

x0 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.25], [0.0, 0.0, 0.5]]).cuda().float()
n0 = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]).cuda().float()

x1 = get_triangle_centers(vertices, triangles).cuda().float()
n1 = get_triangle_normals(vertices, triangles).cuda().float()


def get_error(x, dirichlet):
    rerr = torch.norm(x.real - dirichlet.real) / torch.norm(
        dirichlet.real
    ) + torch.norm(x.imag - dirichlet.imag) / torch.norm(dirichlet.imag)
    return rerr.item() / 2


rerr_dict = {
    "CBIE": [],
    "HBIE": [],
    "Combined": [],
}

for degree in [0, 1]:
    for wave_number in [-1, -10, -100]:
        dirichlet = get_potential_of_sources(x0, x1, n0, n1, wave_number, degree, False)
        neumann = get_potential_of_sources(x0, x1, n0, n1, wave_number, degree, True)

        double_layer_mat = cuda_bem.assemble_boundary_matrix(wave_number, "double")
        single_layer_mat = cuda_bem.assemble_boundary_matrix(wave_number, "single")
        hypersingular_mat = cuda_bem.assemble_boundary_matrix(
            wave_number, "hypersingular"
        )
        adjoint_double_layer_mat = cuda_bem.assemble_boundary_matrix(
            wave_number, "adjointdouble"
        )
        identity_mat = cuda_bem.identity_matrix()

        A = double_layer_mat - 0.5 * identity_mat
        b = single_layer_mat @ neumann
        x = solve_linear_equation(A, b)

        rerr_dict["CBIE"].append(get_error(x, dirichlet))

        A = hypersingular_mat
        b = 0.5 * identity_mat @ neumann + adjoint_double_layer_mat @ neumann
        x = solve_linear_equation(A, b)

        rerr_dict["HBIE"].append(get_error(x, dirichlet))

        beta = 1j / wave_number
        A = (double_layer_mat - 0.5 * identity_mat) + hypersingular_mat * beta
        b = (
            single_layer_mat @ neumann
            + (0.5 * identity_mat @ neumann + adjoint_double_layer_mat @ neumann) * beta
        )

        x = solve_linear_equation(A, b)

        rerr_dict["Combined"].append(get_error(x, dirichlet))


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(rerr_dict["CBIE"], label="CBIE")
ax.plot(rerr_dict["HBIE"], label="HBIE")
ax.plot(rerr_dict["Combined"], label="Combined")
ax.set_xlabel("Test case")
ax.set_ylabel("Relative error")
ax.legend()
plt.show()
