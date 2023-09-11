import sys

sys.path.append("./")
import torch
from src.cuda_imp import ImportanceSampler
import meshio
from src.visualize import get_figure

mesh = meshio.read("data/meshes/cube.obj")

vertices = torch.tensor(mesh.points, dtype=torch.float32).cuda()
triangles = torch.tensor(mesh.cells_dict["triangle"], dtype=torch.int32).cuda()
importance = torch.ones(triangles.shape[0], 1, dtype=torch.float32).cuda()

sampler = ImportanceSampler(vertices, triangles, importance, 1000)
sampler.update()
feature = torch.cat(
    [sampler.points_normals, sampler.points_importance.unsqueeze(-1)], dim=1
)
# get_figure(sampler.points.cpu().numpy(), feature.cpu().numpy()).show()

# triangles_center = vertices[triangles].mean(dim=1)
# importance[triangles_center[:, 2] > 0] = 0.1
# sampler = ImportanceSampler(vertices, triangles, importance, 1000)
# sampler.update()
# feature = torch.cat(
#     [sampler.points_normals, sampler.points_importance.unsqueeze(-1)], dim=1
# )
# get_figure(sampler.points.cpu().numpy(), feature.cpu().numpy()).show()
