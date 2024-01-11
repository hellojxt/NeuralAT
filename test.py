from src.ffat_solve import monte_carlo_solve
from src.visualize import CombinedFig
import torch


data = torch.load("test.pt")

vertices = data["vertices"]
triangles = data["triangles"]
neumann = data["neumann"]
ks = data["ks"]
trg_points = data["trg_points"]
sample_num = data["sample_num"]

CombinedFig().add_mesh(vertices, triangles).show()

monte_carlo_solve(vertices, triangles, neumann, ks, trg_points, sample_num)
