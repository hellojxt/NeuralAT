"""
Compare the boundary matrices computed by Bempp and the CUDA BEM solver.

!!! Note:   Hypersingular operator is not tested here because the implementation 
            of the hypersingular operator in Bempp is different from the CUDA BEM solver.
"""

import sys

sys.path.append("./")
from src.bem.solver import BEM_Solver
import bempp.api
import torch
import numpy as np

torch.set_printoptions(precision=5)

bempp.api.BOUNDARY_OPERATOR_DEVICE_TYPE = "gpu"
bempp.api.POTENTIAL_OPERATOR_DEVICE_TYPE = "gpu"
bempp.api.PLOT_BACKEND = "gmsh"
grid = bempp.api.shapes.regular_sphere(0)
# grid.plot()
vertices = torch.from_numpy(grid.vertices.T.astype("float32")).cuda()
print(vertices)
triangles = torch.from_numpy(grid.elements.T.astype("int32")).cuda()
print(triangles)

cuda_bem = BEM_Solver(vertices, triangles)
space = bempp.api.function_space(grid, "P", 1)

for wave_number in [1, 10, 100]:
    print(f"Wave number: {wave_number}")
    beta = 1j / wave_number
    slp = bempp.api.operators.boundary.helmholtz.single_layer(
        space,
        space,
        space,
        wave_number,
        device_interface="numba",
        precision="single",
    )
    single_matrix_bempp = torch.from_numpy(slp.weak_form().A).cuda()
    single_matrix_cuda = cuda_bem.assemble_boundary_matrix(wave_number, "single")
    rerr = torch.norm(single_matrix_bempp - single_matrix_cuda) / torch.norm(
        single_matrix_bempp
    )
    if rerr > 1e-4:
        print("triangles:", triangles)
        print(f"Relative error single layer: {rerr}")
        print("single_matrix_cuda:")
        print(single_matrix_cuda)
        print("single_matrix_bempp:")
        print(single_matrix_bempp)

    dlp = bempp.api.operators.boundary.helmholtz.double_layer(
        space,
        space,
        space,
        wave_number,
        device_interface="opencl",
        precision="single",
    )
    double_matrix_bempp = torch.from_numpy(dlp.weak_form().A).cuda()
    double_matrix_cuda = cuda_bem.assemble_boundary_matrix(wave_number, "double")

    rerr = torch.norm(double_matrix_bempp - double_matrix_cuda) / torch.norm(
        double_matrix_bempp
    )
    if rerr > 1e-4:
        print("triangles:", triangles)
        print(f"Relative error double layer: {rerr}")
        print("double_matrix_cuda:")
        print(double_matrix_cuda)
        print("double_matrix_bempp:")
        print(double_matrix_bempp)

    adlp = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
        space,
        space,
        space,
        wave_number,
        device_interface="opencl",
        precision="single",
    )

    adjoint_double_matrix_bempp = torch.from_numpy(adlp.weak_form().A).cuda()
    adjoint_double_matrix_cuda = cuda_bem.assemble_boundary_matrix(
        wave_number, "adjointdouble"
    )

    rerr = torch.norm(
        adjoint_double_matrix_bempp - adjoint_double_matrix_cuda
    ) / torch.norm(adjoint_double_matrix_bempp)

    if rerr > 1e-4:
        print("triangles:", triangles)
        print(f"Relative error adjoint double layer: {rerr}")
        print("adjoint_double_matrix_cuda:")
        print(adjoint_double_matrix_cuda)
        print("adjoint_double_matrix_bempp:")
        print(adjoint_double_matrix_bempp)

    hyp = bempp.api.operators.boundary.helmholtz.hypersingular(
        space,
        space,
        space,
        wave_number,
        device_interface="opencl",
        precision="single",
    )

    hyp_matrix_bempp = torch.from_numpy(hyp.weak_form().A).cuda()
    hyp_matrix_cuda = cuda_bem.assemble_boundary_matrix(wave_number, "hypersingular")

    rerr = torch.norm(hyp_matrix_bempp - hyp_matrix_cuda) / torch.norm(hyp_matrix_bempp)

    if rerr > 1e-4:
        print("triangles:", triangles)
        print(f"Relative error hypersingular: {rerr}")
        print("hyp_matrix_cuda:")
        print(hyp_matrix_cuda)
        print("hyp_matrix_bempp:")
        print(hyp_matrix_bempp)

    LHS_bempp = -double_matrix_bempp + beta * hyp_matrix_bempp
    LHS_cuda = cuda_bem.assemble_boundary_matrix(wave_number, "bm_lhs")
    rerr = torch.norm(LHS_bempp - LHS_cuda) / torch.norm(LHS_bempp)
    if rerr > 1e-4:
        print("triangles:", triangles)
        print(f"Relative error LHS: {rerr}")
        print("LHS_cuda:")
        print(LHS_cuda)
        print("LHS_bempp:")
        print(LHS_bempp)

    RHS_bempp = -single_matrix_bempp - beta * adjoint_double_matrix_bempp
    RHS_cuda = cuda_bem.assemble_boundary_matrix(wave_number, "bm_rhs")
    rerr = torch.norm(RHS_bempp - RHS_cuda) / torch.norm(RHS_bempp)
    if rerr > 1e-4:
        print("triangles:", triangles)
        print(f"Relative error RHS: {rerr}")
        print("RHS_cuda:")
        print(RHS_cuda)
        print("RHS_bempp:")
        print(RHS_bempp)
