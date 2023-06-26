import sys

sys.path.append("./")
from src.assemble import (
    assemble_single_boundary_matrix,
    assemble_double_boundary_matrix,
)
import bempp.api
import torch
import numpy as np
import time


def select_two_triangles(grid, i, j):
    triangles_old = grid.elements.T
    triangles_new = triangles_old[[i, j]].T
    print(triangles_new)
    grid_new = bempp.api.Grid(grid.vertices, triangles_new)
    return grid_new


# def square_grid():
#     vertices = np.array(
#         [
#             [0, 0, 0],
#             [1, 0, 0],
#             [0, 1, 0],
#             [1, 1, 0],
#         ]
#     )
#     triangles = np.array([[0, 1, 2], [1, 3, 2]])
#     grid = bempp.api.Grid(vertices.T, triangles.T)
#     return grid


bempp.api.BOUNDARY_OPERATOR_DEVICE_TYPE = "gpu"
bempp.api.POTENTIAL_OPERATOR_DEVICE_TYPE = "gpu"
grid = bempp.api.shapes.sphere(h=0.15)
# grid = select_two_triangles(grid, 48, 63)
# grid = square_grid()
vertices = torch.from_numpy(grid.vertices.T.astype("float32")).cuda()
triangles = torch.from_numpy(grid.elements.T.astype("int32")).cuda()
print(triangles.shape)
# grid.plot()
space = bempp.api.function_space(grid, "DP", 0)
# wave_number = np.random.rand() * 300 + 0.3
for wave_number in [1, 10, 100]:
    print(wave_number)
    start_time = time.time()
    slp = bempp.api.operators.boundary.helmholtz.single_layer(
        space, space, space, wave_number, device_interface="opencl", precision="single"
    )
    single_matrix_bempp = torch.from_numpy(slp.weak_form().A).cuda().real
    t1 = time.time() - start_time
    start_time = time.time()
    single_matrix_cuda = assemble_single_boundary_matrix(
        vertices, triangles, wave_number
    )
    torch.cuda.synchronize()
    t2 = time.time() - start_time
    error = abs(single_matrix_cuda - single_matrix_bempp) / (
        (single_matrix_bempp**2).mean() ** 0.5
    )
    print(
        "single max error: ",
        error.max(),
        " time cost bempp: ",
        t1,
        "time cost cuda: ",
        t2,
    )

    start_time = time.time()
    dlp = bempp.api.operators.boundary.helmholtz.double_layer(
        space, space, space, wave_number, device_interface="numba", precision="single"
    )
    double_matrix_bempp = torch.from_numpy(dlp.weak_form().A).cuda().real
    t1 = time.time() - start_time
    start_time = time.time()
    double_matrix_cuda = assemble_double_boundary_matrix(
        vertices, triangles, wave_number
    )
    t2 = time.time() - start_time
    error = abs(double_matrix_cuda - double_matrix_bempp) / (
        (double_matrix_bempp**2).mean() ** 0.5
    )
    print(
        "double max error: ",
        error.max(),
        " time cost bempp: ",
        t1,
        "time cost cuda: ",
        t2,
    )

# mask = [error == error.max()][0]
# # print the index of trues in mask
# for i in range(triangles.size(0)):
#     for j in range(triangles.size(0)):
#         if mask[i, j]:
#             print(i, j)
# print(error[mask])
# print(double_matrix_bempp[mask])
# print(double_matrix_cuda[mask])
# print(grid.edge_adjacency)
# print(double_matrix_bempp)
# print(double_matrix_cuda)
