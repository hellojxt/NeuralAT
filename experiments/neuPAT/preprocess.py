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
from src.cuda_imp import ImportanceSampler, MonteCarloWeight
from src.visualize import plot_point_cloud, plot_mesh, CombinedFig
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from time import time
import configparser
import meshio
import torch

data_dir = sys.argv[1]

import json

with open(f"{data_dir}/config.json", "r") as file:
    config_data = json.load(file)


def precompute_sample_points():
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

    static_objects = config_data.get("static_obj", [])
    for obj in static_objects:
        mesh = obj.get("mesh")
        size = obj.get("size")
        position = obj.get("position")
        obj_static = MeshObj(f"{data_dir}/{mesh}", scale=size)
        obj_static.vertices = obj_static.vertices + position

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

    vertices_static = torch.from_numpy(obj_static.vertices).cuda().to(torch.float32)
    triangles_static = torch.from_numpy(obj_static.triangles).cuda().to(torch.int32)
    importance_static = torch.ones(len(triangles_static), dtype=torch.float32).cuda()
    sampler_static = ImportanceSampler(
        vertices_static, triangles_static, importance_static, 400000
    )
    sampler_static.update()
    sampler_static.poisson_disk_resample(radius, 8)
    print("static points:", sampler_static.num_samples)

    neumann_static = torch.zeros(
        mode_num, len(sampler_static.points), 1, dtype=torch.complex64
    ).cuda()

    points_static = sampler_static.points
    points_vib = sampler_vib.points
    points_all = torch.cat([points_vib, points_static], dim=0)
    normal_static = sampler_static.points_normals
    normal_vib = sampler_vib.points_normals

    neumann = torch.cat([neumann_vib, neumann_static], dim=1)
    cdf = sampler_vib.cdf[-1] + sampler_static.cdf[-1]
    importance = torch.cat([importance_vib, importance_static], dim=0)
    ks = torch.from_numpy(-np.array(wave_number)).to(torch.float32)
    torch.save(
        {
            "points_static": points_static,
            "points_vib": points_vib,
            "normal_static": normal_static,
            "normal_vib": normal_vib,
            "neumann": neumann,
            "cdf": cdf,
            "importance": importance,
            "ks": ks,
        },
        f"{data_dir}/sample_points.pt",
    )


precompute_sample_points()
