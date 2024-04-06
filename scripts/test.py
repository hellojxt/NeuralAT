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

torch.set_printoptions(precision=5)

mesh_path = "dataset/meshes/spot.obj"
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


# Visualizer().add_mesh(vertices, triangles).add_points(x0).show()

degree = 0
wave_number = -1

dirichlet = get_potential_of_sources(x0, x1, n0, n1, wave_number, degree, False)
neumann = get_potential_of_sources(x0, x1, n0, n1, wave_number, degree, True)

double_layer_mat = cuda_bem.assemble_boundary_matrix(wave_number, "double")
single_layer_mat = cuda_bem.assemble_boundary_matrix(wave_number, "single")
hypersingular_mat = cuda_bem.assemble_boundary_matrix(wave_number, "hypersingular")
adjoint_double_layer_mat = cuda_bem.assemble_boundary_matrix(
    wave_number, "adjointdouble"
)
identity_mat = cuda_bem.identity_matrix()

for check_idx in range(1):

    LHS = 0.5 * identity_mat[check_idx, :] @ dirichlet

    RHS = (
        double_layer_mat[check_idx, :] @ dirichlet
        - single_layer_mat[check_idx, :] @ neumann
    )

    eps = 1e-2
    vertices_update = vertices.clone()
    vertices_update[triangles[check_idx, 0]] += n1[check_idx] * eps
    vertices_update[triangles[check_idx, 1]] += n1[check_idx] * eps
    vertices_update[triangles[check_idx, 2]] += n1[check_idx] * eps

    cuda_bem_update = BEM_Solver(vertices_update, triangles)
    single_layer_mat_update = cuda_bem_update.assemble_boundary_matrix(
        wave_number, "single"
    )
    double_layer_mat_update = cuda_bem_update.assemble_boundary_matrix(
        wave_number, "double"
    )
    x1_update = get_triangle_centers(vertices_update, triangles).cuda().float()
    dirichlet_update = get_potential_of_sources(
        x0, x1_update, n0, n1, wave_number, degree, False
    )
    neumann_update = get_potential_of_sources(
        x0, x1_update, n0, n1, wave_number, degree, True
    )

    LHS_update = 0.5 * identity_mat[check_idx, :] @ dirichlet_update

    LHS_approx = (LHS_update - LHS) / eps
    LHS_cal = 0.5 * identity_mat[check_idx, :] @ neumann

    RHS_update = (
        double_layer_mat_update[check_idx, :] @ dirichlet_update
        - single_layer_mat_update[check_idx, :] @ neumann_update
    )
    print("LHS:", LHS, " | RHS:", RHS)
    print("LHS Update:", LHS_update, " | RHS Update:", RHS_update)
    RHS_approx = (RHS_update - RHS) / eps

    RHS_cal = (
        hypersingular_mat[check_idx, :] @ dirichlet
        - adjoint_double_layer_mat[check_idx, :] @ neumann
    )
    print("LHS Approx:", LHS_approx, " | LHS Cal:", LHS_cal)
    print("RHS Approx:", RHS_approx, " | RHS Cal:", RHS_cal)

    RHS_double = double_layer_mat[check_idx, :] * dirichlet
    RHS_update_double = double_layer_mat_update[check_idx, :] * dirichlet_update

    RHS_double_approx = (RHS_update_double - RHS_double) / eps

    RHS_double_cal = hypersingular_mat[check_idx, :] * dirichlet

    for j in range(100):
        print(RHS_double_approx[j], RHS_double_cal[j])
