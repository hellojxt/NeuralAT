import sys

sys.path.append("./")

from src.modalobj.model import (
    ModalSoundObj,
    MatSet,
    Material,
    BEMModel,
    MeshObj,
    get_spherical_surface_points,
)
from src.mcs.mcs import ImportanceSampler, MonteCarloWeight
from src.utils import plot_point_cloud, plot_mesh, CombinedFig
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from time import time
import configparser
import meshio
import torch

data_dir = "dataset/NeuPAT/audio"

import json

with open(f"{data_dir}/config.json", "r") as file:
    config_data = json.load(file)


def precompute_sample_points():
    vibration_objects = config_data.get("vibration_obj", [])
    for obj in vibration_objects:
        mesh = obj.get("mesh")
        size = obj.get("size")
        obj_vib = MeshObj(f"{data_dir}/{mesh}", scale=size)

    static_objects = config_data.get("static_obj", [])
    for obj in static_objects:
        mesh = obj.get("mesh")
        size = obj.get("size")
        obj_static = MeshObj(f"{data_dir}/{mesh}", scale=size)

    radius = config_data.get("solver").get("radius")
    importance_vib = torch.ones(len(obj_vib.triangles), dtype=torch.float32).cuda()
    vertices_vib = torch.from_numpy(obj_vib.vertices).cuda().to(torch.float32)
    triangles_vib = torch.from_numpy(obj_vib.triangles).cuda().to(torch.int32)
    sampler_vib = ImportanceSampler(vertices_vib, triangles_vib, importance_vib, 400000)
    sampler_vib.update()
    sampler_vib.poisson_disk_resample(radius, 8)
    print("vibration points:", sampler_vib.num_samples)

    neumann_vib = torch.ones(len(sampler_vib.points), 1, dtype=torch.complex64).cuda()

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
        len(sampler_static.points), 1, dtype=torch.complex64
    ).cuda()

    points_static = sampler_static.points
    points_vib = sampler_vib.points
    points_all = torch.cat([points_vib, points_static], dim=0)
    normal_static = sampler_static.points_normals
    normal_vib = sampler_vib.points_normals

    neumann = torch.cat([neumann_vib, neumann_static], dim=0)
    cdf = sampler_vib.cdf[-1] + sampler_static.cdf[-1]
    importance = torch.cat([importance_vib, importance_static], dim=0)

    CombinedFig().add_points(points_all, neumann.real).show()

    torch.save(
        {
            "vertices_vib": vertices_vib,
            "triangles_vib": triangles_vib,
            "vertices_static": vertices_static,
            "triangles_static": triangles_static,
            "neumann_vib": neumann_vib,
            "neumann_static": neumann_static,
            "points_static": points_static,
            "points_vib": points_vib,
            "normal_static": normal_static,
            "normal_vib": normal_vib,
            "neumann": neumann,
            "cdf": cdf,
            "importance": importance,
        },
        f"{data_dir}/sample_points.pt",
    )


precompute_sample_points()
