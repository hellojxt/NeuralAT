import sys

sys.path.append("./")

import torch
from src.cuda_imp import ImportanceSampler, MonteCarloWeight
from src.loader.model import ModalSoundObject
import numpy as np
from src.net import get_mlps
from src.visualize import plot_mesh, plot_point_cloud
import os
import time

sound_object = ModalSoundObject(f"dataset/00003")
vertices = torch.tensor(sound_object.vertices, dtype=torch.float32).cuda()
triangles = torch.tensor(sound_object.triangles, dtype=torch.int32).cuda()
importance = torch.ones(len(triangles), dtype=torch.float32).cuda()
sampler = ImportanceSampler(vertices, triangles, importance, 500000)

sampler.update()
points = sampler.points
mask = sampler.poisson_disk_resample(0.003, 1).bool()

print(mask)
print(mask.sum())
print(mask.max())
print(points.shape, mask.shape)
print(points[mask].shape)
plot_point_cloud(
    sound_object.vertices, sound_object.triangles, points[mask].cpu().numpy()
).show()

sampler = ImportanceSampler(vertices, triangles, importance, 21350)
sampler.update()
points = sampler.points
plot_point_cloud(
    sound_object.vertices, sound_object.triangles, points.cpu().numpy()
).show()
