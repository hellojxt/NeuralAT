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

# warm up
sampler = ImportanceSampler(vertices, triangles, importance, 500000)
sampler.update()
sampler.poisson_disk_resample(0.003, 4)

# start test
start_time = time.time()
sampler = ImportanceSampler(vertices, triangles, importance, 500000)
sampler.update()
sampler.poisson_disk_resample(0.002, 4)
print("sample points: ", sampler.num_samples)
print("possion disk sampling cost time: ", time.time() - start_time)

# compare
plot_point_cloud(sound_object.vertices, sound_object.triangles, sampler.points).show()

start_time = time.time()
sampler = ImportanceSampler(vertices, triangles, importance, sampler.num_samples)
sampler.update()
print("random sampling cost time: ", time.time() - start_time)
points = sampler.points
plot_point_cloud(sound_object.vertices, sound_object.triangles, points).show()
