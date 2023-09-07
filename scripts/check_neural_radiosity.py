import sys

sys.path.append("./")
import torch
import meshio
from src.visualize import get_figure

points = torch.load("output/trg.pt")
gt = torch.load("output/gt.pt")
LHS = torch.load("output/LHS.pt")
feature = torch.cat([gt, LHS], dim=1)

get_figure(points.detach().cpu().numpy(), feature.detach().cpu().numpy()).show()
