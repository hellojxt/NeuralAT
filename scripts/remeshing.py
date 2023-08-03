import pymeshlab
import time
import os
import shutil
from tqdm import tqdm
import numpy as np


def remesh_single_obj(
    input_path="abc.obj", output_path="abc_2.obj", iterations=8, percentage=3.0, normalize=False
):
    # remesh a single objecct so that the area of each face is roughly the same
    # not supported on arm macs
    # iterations: number of iterations, higher iter will result in more uniform mesh but longer time
    # percentage: average length of the output edges. 3 means 3%, and will lead to about 1~2k vertices in the output mesh
    # additionally, this script will normalize the mesh into the cube of [-1,1]^3

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_path)
    # target_len = ms.compute_average_edge_length()
    target_len = pymeshlab.Percentage(percentage)
    ms.apply_filter(
        "meshing_isotropic_explicit_remeshing",
        iterations=iterations,
        targetlen=target_len,
    )
    if normalize:
        # ...normalize the mesh into [-1, 1]^3
        scale_x = ms.current_mesh().vertex_matrix()[:, 0].max() - ms.current_mesh().vertex_matrix()[:, 0].min()
        scale_y = ms.current_mesh().vertex_matrix()[:, 1].max() - ms.current_mesh().vertex_matrix()[:, 1].min()
        scale_z = ms.current_mesh().vertex_matrix()[:, 2].max() - ms.current_mesh().vertex_matrix()[:, 2].min()
        scale_all = max(scale_x, scale_y, scale_z)
        ms.apply_filter('compute_matrix_from_scaling_or_normalization', axisx=scale_all, axisy=scale_all, axisz=scale_all)
    
    ms.save_current_mesh(output_path, save_vertex_normal=False, save_textures=False)


if __name__ == "__main__":
    IN_PATH = "dataset/ABC_Dataset/surf_mesh"
    OUT_PATH = "dataset/ABC_Dataset/surf_mesh_remeshed"
    if os.path.exists(IN_PATH):
        if os.path.exists(OUT_PATH):
            shutil.rmtree(OUT_PATH)
            os.mkdir(OUT_PATH)
        else:
            os.mkdir(OUT_PATH)
        all_files = os.listdir(IN_PATH)
        all_files = [file for file in all_files if file.endswith(".obj")]
        for file in tqdm(all_files):
            in_name, out_name = os.path.join(IN_PATH, file), os.path.join(
                OUT_PATH, file
            )
            # continue if the in_name is an empty file
            with open(in_name, "r") as f:
                if len(f.readlines()) < 100:
                    continue
            try:
                remesh_single_obj(in_name, out_name)
            except Exception as e:
                # very rare cases
                print("Failed to remesh", in_name, "due to", e)

    print("Done!")
