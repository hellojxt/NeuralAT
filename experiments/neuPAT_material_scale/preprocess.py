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
from src.mcs.mcs import ImportanceSampler
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

vertices_vib = torch.from_numpy(obj_vib.surf_vertices).cuda().to(torch.float32)
triangles_vib = torch.from_numpy(obj_vib.surf_triangles).cuda().to(torch.int32)

neumann_tri = np.zeros((mode_num, len(triangles_vib)), dtype=np.complex64)
wave_number = []
for i in range(mode_num):
    neumann_tri[i] = obj_vib.get_triangle_neumann(i)
    wave_number.append(-obj_vib.get_wave_number(i))

import meshio
import os

meshio.write_points_cells(
    os.path.join(data_dir, "mesh_surf.obj"),
    vertices_vib.cpu().numpy(),
    [("triangle", triangles_vib.cpu().numpy())],
)

neumann_tri = torch.from_numpy(neumann_tri).cuda()
ks = torch.tensor(wave_number).to(torch.float32).cuda()
torch.save(
    {
        "vertices": vertices_vib,
        "triangles": triangles_vib,
        "neumann_tri": neumann_tri,
        "ks": ks,
        "eigenvalues": obj_vib.eigenvalues,
        "modes": obj_vib.modes,
    },
    f"{data_dir}/modal_data.pt",
)
