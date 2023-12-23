import sys

sys.path.append("./")

from src.modalsound.model import (
    SNR,
)
import numpy as np

import matplotlib.pyplot as plt
from glob import glob


root_dir = "dataset/monopole"

obj_dir = glob(f"{root_dir}/*")

for obj_path in obj_dir:
    gt_data = np.load(obj_path + "/gt.npz")
    bem_data = np.load(obj_path + "/bem.npz")
    NeuralSound_data = np.load(obj_path + "/NeuralSound.npz")
    ours_data = np.load(obj_path + "/ours.npz")

    gt_ffat = gt_data["ffat_map"].reshape(6, 64, 32)
    bem_ffat = bem_data["ffat_map"].reshape(*gt_ffat.shape)
    NeuralSound_ffat = NeuralSound_data["ffat_map"].reshape(*gt_ffat.shape) / 0.15
    ours_ffat = ours_data["ffat_map"].reshape(*gt_ffat.shape)

    bem_time = bem_data["cost_time"]
    ours_time = ours_data["cost_time"]
    NeuralSound_time = (
        NeuralSound_data["cost_time"] + np.load(obj_path + "/voxel.npz")["cost_time"]
    )
    print(NeuralSound_data["cost_time"], np.load(obj_path + "/voxel.npz")["cost_time"])

    for i in range(6):
        print(
            f"{obj_path.split('/')[-1]}_{i}: bem SNR: {SNR(gt_ffat[i], bem_ffat[i])}, time: {bem_time}"
        )
        print(
            f"{obj_path.split('/')[-1]}_{i}: ours SNR: {SNR(gt_ffat[i], ours_ffat[i])}, time: {ours_time}"
        )
        print(
            f"{obj_path.split('/')[-1]}_{i}: NeuralSound SNR: {SNR(gt_ffat[i], NeuralSound_ffat[i])}, time: {NeuralSound_time}"
        )

    # for i in range(3, 6):
    #     plt.title(f"{obj_path.split('/')[-1]}_{i}")
    #     plt.subplot(2, 2, 1)
    #     plt.imshow(gt_ffat[i].real, label="gt")
    #     plt.colorbar()
    #     plt.subplot(2, 2, 2)
    #     plt.imshow(bem_ffat[i].real, label="bem")
    #     plt.colorbar()
    #     plt.subplot(2, 2, 3)
    #     plt.imshow(NeuralSound_ffat[i], label="NeuralSound")
    #     plt.colorbar()
    #     plt.subplot(2, 2, 4)
    #     plt.imshow(ours_ffat[i].real, label="ours")
    #     plt.colorbar()
    #     plt.show()
    #     plt.close()
