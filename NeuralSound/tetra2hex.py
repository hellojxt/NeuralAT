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
from src.visualize import plot_point_cloud
from src.modalsound.model import MeshObj
from scipy.spatial import KDTree

freqs = [20, 200, 2000, 20, 200, 2000]

root_dir = sys.argv[1]
obj_dir = glob(f"{root_dir}/*")
for obj_path in obj_dir:
    mesh = MeshObj(obj_path + "/mesh.obj")
    data = np.load(obj_path + "/bem.npz")
    points = mesh.triangles_center
    normals = mesh.triangles_normal
    neumann = data["neumann"].reshape(-1, len(points)).T
    neumann = neumann.reshape(len(points), 1, -1)
    normals = normals.reshape(len(points), 3, 1)
    neumann = neumann * normals

    hexa = Hexa_model(mesh.vertices, mesh.triangles, res=32)
    hexa.fill_shell()
    voxel = hexa.voxel_grid
    coords = to_sparse_coords(voxel).astype(np.float32)
    coords = ((coords + 0.5) / 32 - 0.5) * 0.15

    kd_tree = KDTree(points)
    _, idx = kd_tree.query(coords)
    vecs = neumann[idx]

    coords = to_sparse_coords(voxel)
    coords_surface, feats_index = map(np.asarray, boundary_voxel(coords))
    surface_code = np.asarray(boundary_encode(coords_surface))
    image_size = 32
    mode_num = len(freqs)
    ffat_map, ffat_map_far = np.zeros((mode_num, 2 * image_size, image_size)), np.zeros(
        (mode_num, 2 * image_size, image_size)
    )
    np.savez_compressed(
        obj_path + "/voxel.npz",
        coords=coords_surface,
        feats_in=vecs[feats_index],
        feats_out=ffat_map,
        feats_out_far=ffat_map_far,
        surface=surface_code,
        freqs=freqs,
    )
