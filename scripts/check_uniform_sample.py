import sys

sys.path.append("./")
import torch
from src.cuda_imp import UniformSampler
import meshio
from src.visualize import get_figure

mesh = meshio.read("data/meshes/cube.obj")

vertices = torch.tensor(mesh.points, dtype=torch.float32).cuda()
triangles = torch.tensor(mesh.cells_dict["triangle"], dtype=torch.int32).cuda()

uniform_sampler = UniformSampler(vertices, triangles, 1000)
points, points_normals, inv_pdf = uniform_sampler.update()
print(inv_pdf.mean())
get_figure(points.cpu().numpy(), points_normals.cpu().numpy()).show()
