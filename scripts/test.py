import sys
sys.path.append("./")
import commentjson as json
import tinycudann as tcnn
import torch

with open("data/config.json") as f:
	config = json.load(f)
	
model = tcnn.NetworkWithInputEncoding(
	3, 1,
	config["encoding"], config["network"]
)

from src.sampler import uniform_sample
import meshio

mesh = meshio.read("data/meshes/cube.obj")


vertices = torch.tensor(mesh.points, dtype=torch.float32).cuda()
triangles = torch.tensor(mesh.cells_dict["triangle"], dtype=torch.int32).cuda()

points, points_normals = uniform_sample(vertices, triangles, 1000)


