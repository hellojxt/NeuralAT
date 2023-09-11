import sys

sys.path.append("./")
import torch
import meshio
from src.visualize import get_figure

points = torch.load("output/points.pt")
gt = torch.load("output/gt.pt")
LHS = torch.load("output/LHS.pt")
RHS = torch.load("output/RHS.pt")
feature = torch.cat([gt, LHS, RHS], dim=1)

get_figure(points.detach().cpu().numpy(), feature.detach().cpu().numpy()).show()
