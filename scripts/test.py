import sys

sys.path.append("./")
from loader.model import read_mode_data, load_ffat_map
from src.visualize import get_figure
import meshio
import numpy as np

nDOF, nModes, omega_squared, modes = read_mode_data("dataset/0/00000_surf.modes")
print("nDOF: ", nDOF)
print("nModes: ", nModes)
print("omega_squared: ", omega_squared.shape)
print("modes: ", modes.shape)

mesh = meshio.read("dataset/0/00000.tet.obj")
vertices = mesh.points
triangles = mesh.cells_dict["triangle"]
print("vertices: ", vertices.shape)
print("triangles: ", triangles.shape)

modes = modes.T.reshape(vertices.shape[0], 3, nModes)
modes = (modes**2).sum(axis=1)
print("amp modes: ", modes.shape)

# get_figure(vertices, modes).show()

(
    cell_size,
    low_corners,
    n_elements,
    strides,
    center,
    bbox_low,
    bbox_top,
    k,
    center_3,
    is_compressed,
    psi,
    mode_id,
) = load_ffat_map("dataset/0/00000_ffat_maps/numerical_fdtd-0.fatcube")
print("cell_size: ", cell_size)
print("low_corners: ", low_corners)
print("n_elements: ", n_elements)
print("strides: ", strides)
print("center: ", center)
print("bbox_low: ", bbox_low)
print("bbox_top: ", bbox_top)
print("k: ", k)
print("center_3: ", center_3)
print("is_compressed: ", is_compressed)
# print("psi: ", psi)
print("mode_id: ", mode_id)
psi = np.array(psi)
print("psi: ", psi.shape)
import matplotlib.pyplot as plt

plt.imshow(psi[0])
plt.show()
