import numpy as np
import sys


data_dir = sys.argv[-1]


import meshio

data = np.load(f"{data_dir}/bem.npz")
vertices = data["vertices"]
triangles = data["triangles"]
import os

meshio.write_points_cells(
    os.path.join(data_dir, "mesh_surf.obj"),
    vertices,
    [("triangle", triangles)],
)
