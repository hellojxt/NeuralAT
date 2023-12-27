import sys

sys.path.append("./")

from src.modalsound.model import (
    ModalSoundObj,
    MatSet,
    Material,
    BEMModel,
    MeshObj,
    get_spherical_surface_points,
)
from src.cuda_imp import ImportanceSampler
import numpy as np
import torch

data_dir = "dataset/NeuPAT/scale"

import json

with open(f"{data_dir}/config.json", "r") as file:
    config_data = json.load(file)

vibration_objects = config_data.get("vibration_obj", [])
for obj in vibration_objects:
    mesh = obj.get("mesh")
    size = obj.get("size")
    material = obj.get("material")
    mode_num = obj.get("mode_num")
    obj_vib = ModalSoundObj(f"{data_dir}/{mesh}")
    obj_vib.normalize(size)
    obj_vib.modal_analysis(k=mode_num, material=Material(getattr(MatSet, material)))
    print(obj_vib.get_frequencies())

radius = config_data.get("solver").get("radius")
importance_vib = torch.ones(len(obj_vib.surf_triangles), dtype=torch.float32).cuda()
vertices_vib = torch.from_numpy(obj_vib.surf_vertices).cuda().to(torch.float32)
triangles_vib = torch.from_numpy(obj_vib.surf_triangles).cuda().to(torch.int32)
sampler_vib = ImportanceSampler(vertices_vib, triangles_vib, importance_vib, 400000)
sampler_vib.update()
sampler_vib.poisson_disk_resample(radius, 8)
print("vibration points:", sampler_vib.num_samples)

neumann = np.zeros((mode_num, len(triangles_vib)), dtype=np.complex64)
wave_number = []
for i in range(mode_num):
    neumann[i] = obj_vib.get_triangle_neumann(i)
    wave_number.append(obj_vib.get_wave_number(i))

neumann_vib = torch.from_numpy(neumann).unsqueeze(-1).cuda()
neumann_vib = neumann_vib[:, sampler_vib.points_index, :]
points_vib = sampler_vib.points
normal_vib = sampler_vib.points_normals
cdf = sampler_vib.cdf[-1]
ks = torch.from_numpy(-np.array(wave_number)).to(torch.float32)
torch.save(
    {
        "vertices": vertices_vib,
        "triangles": triangles_vib,
        "neumann_tri": torch.from_numpy(neumann),
        "points_vib": points_vib,
        "normal_vib": normal_vib,
        "neumann": neumann_vib,
        "cdf": cdf,
        "importance": importance_vib,
        "ks": ks,
    },
    f"{data_dir}/sample_points.pt",
)
