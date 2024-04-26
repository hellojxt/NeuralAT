import sys

sys.path.append("./NeuralSound/")
sys.path.append("./")

import numpy as np
import os
from glob import glob
from tqdm import tqdm
from classic.voxelize.hexahedral import Hexa_model
from classic.fem.util import to_sparse_coords
from classic.bem.util import boundary_encode, boundary_voxel
from src.utils import CombinedFig
from src.modalobj.model import MeshObj
from scipy.spatial import KDTree
from torch_geometric.nn.unpool import knn_interpolate
from time import time
import configparser
import torch

root_dir = sys.argv[1]
obj_dir = glob(f"{root_dir}/*")


def update_normals(vertices, triangles):
    """
    vertices: (n, 3)
    triangles: (m, 3)
    """
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return normals


def normalize_vertices(vertices, size=1.0):
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    vertices = (
        (vertices - (bbox_max + bbox_min) / 2) / (bbox_max - bbox_min).max() * size
    )
    return vertices


def get_mesh_size(vertices):
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    return (bbox_max - bbox_min).max()


def run():
    for obj_path in obj_dir:
        if not os.path.isdir(obj_path):
            continue
        data = np.load(obj_path + "/bem.npz")
        vertices, triangles = data["vertices"], data["triangles"]
        vertices = normalize_vertices(vertices)

        triangles_center = vertices[triangles].mean(axis=1)
        normals = update_normals(vertices, triangles).reshape(1, -1, 3)
        wave_number = np.abs(data["wave_number"])
        freqs = wave_number * 343.2 / (2 * np.pi)

        mode_num = len(wave_number)
        neumann = data["neumann"].reshape(mode_num, -1, 1)
        neumann = neumann * normals

        neumann = torch.from_numpy(neumann)
        neumann = neumann.permute(1, 2, 0).cpu().numpy()

        start_time = time()

        hexa = Hexa_model(vertices, triangles, res=32)
        hexa.fill_shell()
        voxel = hexa.voxel_grid
        coords = to_sparse_coords(voxel).astype(np.float32)
        coords = (coords + 0.5) / 32 - 0.5

        kd_tree = KDTree(triangles_center)
        _, idx = kd_tree.query(coords)
        vecs = neumann[idx]
        if vecs.dtype == np.complex64 or vecs.dtype == np.complex128:
            vecs = vecs.real.astype(np.float32)
        cost_time = time() - start_time
        image_size = 32

        coords = to_sparse_coords(voxel)
        coords_surface, feats_index = map(np.asarray, boundary_voxel(coords))
        surface_code = np.asarray(boundary_encode(coords_surface))
        mode_num = len(freqs)
        ffat_map, ffat_map_far = np.zeros(
            (mode_num, 2 * image_size, image_size)
        ), np.zeros((mode_num, 2 * image_size, image_size))
        print(cost_time)
        out_path = obj_path + "/voxel.npz"
        np.savez_compressed(
            out_path,
            coords=coords_surface,
            feats_in=vecs[feats_index],
            feats_out=ffat_map,
            feats_out_far=ffat_map_far,
            surface=surface_code,
            freqs=freqs,
            cost_time=cost_time,
        )


run()
run()
