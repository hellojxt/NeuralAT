import numpy as np
import sys


root_dir = sys.argv[-1]


import meshio
import os
from glob import glob

for data_dir in glob(f"{root_dir}/*"):
    if not os.path.isdir(data_dir):
        continue
    print(data_dir)
    data = np.load(f"{data_dir}/bem.npz")
    vertices = data["vertices"]
    triangles = data["triangles"]
    import os

    meshio.write_points_cells(
        os.path.join(data_dir, "mesh_surf.obj"),
        vertices,
        [("triangle", triangles)],
    )
