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

    hyp = bempp.api.operators.boundary.helmholtz.hypersingular(
        space,
        space,
        space,
        wave_number,
        device_interface="numba",
        precision="single",
    )
    hyp_matrix_bempp = torch.from_numpy(hyp.weak_form().A).cuda()

    hyp_singular = bempp.api.operators.boundary.helmholtz.hypersingular(
        space,
        space,
        space,
        wave_number,
        assembler="only_singular_part",
        device_interface="numba",
        precision="single",
    )
    hyp_singular_matrix_bempp = torch.from_numpy(
        hyp_singular.weak_form().A.todense()
    ).cuda()
    hyp_regular_matrix_bempp = hyp_matrix_bempp - hyp_singular_matrix_bempp
    hyp_regular_matrix_cuda, hyp_singular_matrix_cuda = (
        cuda_bem.assemble_boundary_matrix(wave_number, "hypersingular")
    )
    print(hyp_regular_matrix_cuda)
    print(hyp_regular_matrix_bempp)

    print(hyp_singular_matrix_cuda)
    print(hyp_singular_matrix_bempp)
    break
