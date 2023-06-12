from src.assemble import boundary_operator_assembler
import bempp.api
import torch
import numpy as np


def select_two_triangles(grid, i, j):
    triangles_old = grid.elements.T
    triangles_new = triangles_old[[i, j]].T
    grid_new = bempp.api.Grid(grid.vertices, triangles_new)
    return grid_new


bempp.api.BOUNDARY_OPERATOR_DEVICE_TYPE = "gpu"
bempp.api.POTENTIAL_OPERATOR_DEVICE_TYPE = "gpu"
grid = bempp.api.shapes.sphere(h=0.7)
grid = select_two_triangles(grid, 7, 8)
vertices = torch.from_numpy(grid.vertices.T.astype("float32")).cuda()
triangles = torch.from_numpy(grid.elements.T.astype("int32")).cuda()

space = bempp.api.function_space(grid, "DP", 0)
wave_number = 30
slp = bempp.api.operators.boundary.helmholtz.single_layer(
    space, space, space, wave_number, device_interface="numba", precision="single"
)
single_matrix_bempp = torch.from_numpy(slp.weak_form().A).cuda().real
single_matrix_cuda = torch.zeros(triangles.shape[0], triangles.shape[0]).cuda()
boundary_operator_assembler(vertices, triangles, single_matrix_cuda, wave_number, False)
error = (
    (single_matrix_cuda - single_matrix_bempp) ** 2 / single_matrix_bempp**2
) ** 0.5
print("single max error: ", error.max())


# dlp = bempp.api.operators.boundary.helmholtz.double_layer(
#     space, space, space, wave_number, device_interface="numba", precision="single"
# )
# double_matrix_bempp = torch.from_numpy(dlp.weak_form().A).cuda().real
# double_matrix_cuda = torch.zeros(triangles.shape[0], triangles.shape[0]).cuda()
# boundary_operator_assembler(vertices, triangles, double_matrix_cuda, wave_number, True)
# error = (
#     (double_matrix_cuda - double_matrix_bempp) ** 2 / double_matrix_bempp**2
# ) ** 0.5
# print("double max error: ", error.max())

mask = [error == error.max()][0]
# print the index of trues in mask
for i in range(triangles.size(0)):
    for j in range(triangles.size(0)):
        if mask[i, j]:
            print(i, j)
print(error[mask])
print(single_matrix_bempp[mask])
print(single_matrix_cuda[mask])
print(grid.edge_adjacency)
