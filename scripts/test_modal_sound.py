import sys

sys.path.append("./")

from src.modalsound.model import SoundObj
from src.visualize import plot_point_cloud, plot_mesh
import numpy as np

obj = SoundObj("dataset/00002/00002.tet.obj")
obj.modal_analysis()

print(obj.get_frequency(0))

data = (obj.modes**2).sum(1)[:, :4]
plot_mesh(obj.surf_vertices, obj.surf_triangles, data).show()
plot_mesh(
    obj.surf_vertices, obj.surf_triangles, obj.get_triangle_neumanns([0, 1, 2, 3])
).show()

bm = obj.solve_by_BEM(0)

plot_mesh(
    obj.surf_vertices, obj.surf_triangles, np.abs(bm.get_dirichlet_coeff())
).show()
