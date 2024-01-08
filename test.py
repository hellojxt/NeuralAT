import numpy as np
from glob import glob
import torch


data_dir = "dataset/NeuPAT/bowl"
data_list = glob(f"{data_dir}/data_*.pt")
xs = []
ys = []
for data_points in data_list:
    data = torch.load(data_points)
    print(data_points)
    x, y = data["x"], data["y"]
    print((x).max(), (y).max())
    print((x).min(), (y).min())
    print(x.shape, y.shape)
    xs.append(x)
    ys.append(y)
