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
from scipy.spatial import KDTree

root_dir = "dataset/ABC"

eigen_dir = os.path.join(root_dir, "data/eigen")

eigen_list = glob(os.path.join(eigen_dir, "*.npz"))

for eigen_path in tqdm(eigen_list):
    out_path = eigen_path.replace("eigen", "acoustic")
    if os.path.exists(out_path):
        continue
    data = np.load(eigen_path)
    vertices, triangles = data["vertices"], data["triangles"]
    wave_number = data["wave_number"]
    modes = data["modes"]
    freqs = wave_number * 343.2 / (2 * np.pi)
    hexa = Hexa_model(vertices / 0.15, triangles, res=32)
    hexa.fill_shell()
    voxel = hexa.voxel_grid
    coords = to_sparse_coords(voxel).astype(np.float32)
    coords = ((coords + 0.5) / 32 - 0.5) * 0.15

    kd_tree = KDTree(vertices)
    _, idx = kd_tree.query(coords)
    vecs = modes[idx]

    # vertex_feats = modes[:, 0, 0]
    # plot_point_cloud(vertices, triangles, vertices, vertex_feats).show()
    # voxel_feats = vecs[:, 0, 0]
    # plot_point_cloud(vertices, triangles, coords, voxel_feats).show()

    coords = to_sparse_coords(voxel)
    coords_surface, feats_index = map(np.asarray, boundary_voxel(coords))
    surface_code = np.asarray(boundary_encode(coords_surface))
    image_size = 32
    mode_num = len(freqs)
    ffat_map, ffat_map_far = np.zeros((mode_num, 2 * image_size, image_size)), np.zeros(
        (mode_num, 2 * image_size, image_size)
    )
    np.savez_compressed(
        out_path,
        coords=coords_surface,
        feats_in=vecs[feats_index],
        feats_out=ffat_map,
        feats_out_far=ffat_map_far,
        surface=surface_code,
        freqs=freqs,
    )
