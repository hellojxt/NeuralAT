import sys

sys.path.append("./")

import torch
from src.cuda_imp import FDTDSimulator, get_bound_info
from src.visualize import CombinedFig
from src.modalsound.model import MeshObj
from src.timer import Timer
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

res = 64
freq = 8000
size = 0.3
obj = MeshObj("dataset/sphere.obj", 0.1)
vertices = torch.from_numpy(obj.vertices).cuda().float()
triangles = torch.from_numpy(obj.triangles).cuda().int()
min_bound, max_bound, bound_size = get_bound_info(obj.vertices, padding=3)

simulation_time = 0.1
listen_pos = torch.tensor([0.0, 0.0, 0.1], device="cuda", dtype=torch.float32)
fdtd = FDTDSimulator(min_bound, max_bound, bound_size, res, listen_pos)

batch_size = 500
full_step_num = int(simulation_time / fdtd.dt)
full_step_num = full_step_num - full_step_num % batch_size

ts = torch.arange(full_step_num, device="cuda", dtype=torch.float32) * fdtd.dt
signal = torch.zeros(full_step_num + batch_size, device="cuda", dtype=torch.float32)

for t_i in tqdm(range(0, full_step_num, batch_size)):
    ts_batch = ts[t_i : t_i + batch_size]
    timer = Timer()
    triangles_neumann = (
        torch.sin(2 * np.pi * freq * ts_batch)
        .unsqueeze(1)
        .repeat(1, triangles.shape[0])
        .T.cuda()
        .float()
    )
    print("generate signal", timer.get_time())
    fdtd.reset_grid()
    batch_signal = fdtd.update(vertices, triangles, triangles_neumann)
    signal[t_i : t_i + batch_size * 2] += batch_signal

plt.subplot(211)
plt.plot(triangles_neumann[0].cpu().numpy())
plt.subplot(212)
plt.plot(signal.cpu().numpy())
plt.show()
plt.close()
