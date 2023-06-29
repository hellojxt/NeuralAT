import sys

sys.path.append("./")
from src.mesh_loader import tetra_surf_from_triangle_mesh
from glob import glob
from tqdm import tqdm


mesh_dir = sys.argv[1]
out_surf_mesh_dir = sys.argv[2]
import os

if not os.path.exists(out_surf_mesh_dir):
    os.makedirs(out_surf_mesh_dir)
for mesh_file in tqdm(glob(mesh_dir + "/*.obj")):
    tetra_surf_from_triangle_mesh(mesh_file, out_surf_mesh_dir)
