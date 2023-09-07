import sys
sys.path.append("./")
import torch
from src.sampler import uniform_sample
import meshio

mesh = meshio.read("data/meshes/cube.obj")


vertices = torch.tensor(mesh.points, dtype=torch.float32).cuda()
triangles = torch.tensor(mesh.cells_dict["triangle"], dtype=torch.int32).cuda()

points, points_normals = uniform_sample(vertices, triangles, 1000)


from src.visualize import get_figure

get_figure(points.cpu().numpy(), points_normals.cpu().numpy()).show()
