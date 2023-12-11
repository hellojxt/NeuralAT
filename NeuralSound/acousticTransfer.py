import sys

sys.path.append("./")
from classic.fem.util import to_sparse_coords, vertex_to_voxel_matrix
from classic.bem.ffat import ffat_to_imgs, vibration_to_ffat
from classic.bem.util import boundary_encode, boundary_voxel
from classic.tools import freq2mel, mel2freq, index2mel, MelConfig
from classic.fem.femModel import Material, random_material
from tqdm import tqdm
import numpy as np
import os
from glob import glob
import bempp.api

bempp.api.enable_console_logging("debug")


def process_single_model(filename, output_name):
    dir_name = os.path.dirname(output_name)
    os.makedirs(dir_name, exist_ok=True)
    data = np.load(filename)
    voxel, vecs, freqs = data["voxel"], data["vecs"], data["freqs"]
    coords = to_sparse_coords(voxel)
    coords_surface, feats_index = map(np.asarray, boundary_voxel(coords))
    surface_code = np.asarray(boundary_encode(coords_surface))
    vertex_num = vecs.shape[0] // 3
    mode_num = len(freqs)
    idx_lst = np.arange(mode_num)
    freqs_lst = []
    vecs_lst = []

    for mode_idx in idx_lst[:1]:
        vec = vecs[:, mode_idx].reshape(vertex_num, -1)
        vec = vertex_to_voxel_matrix(voxel) @ vec
        freq = freqs[mode_idx]
        freqs_lst.append(freq)
        vecs_lst.append(vec)

    vecs = np.stack(vecs_lst, axis=2)
    freqs = np.array(freqs_lst)
    # print(vecs.shape, freqs.shape)
    image_size = 32

    ffat_map, ffat_map_far = vibration_to_ffat(
        coords, vecs, freqs, image_size=image_size
    )

    # ffat_map, ffat_map_far = np.zeros((mode_num, 2 * image_size, image_size)), np.zeros(
    #     (mode_num, 2 * image_size, image_size)
    # )

    # ffat_to_imgs(ffat_map, "./", tag="ffat")
    # ffat_to_imgs(ffat_map_far, "./", tag="ffat_far")

    np.savez_compressed(
        output_name,
        coords=coords_surface,
        feats_in=vecs[feats_index],
        feats_out=ffat_map,
        feats_out_far=ffat_map_far,
        surface=surface_code,
        freqs=freqs,
    )


if __name__ == "__main__":
    file_list = glob(sys.argv[1])
    for filename in tqdm(file_list):
        out_name = filename.replace("eigenData", "acousticData")
        process_single_model(filename, out_name)
