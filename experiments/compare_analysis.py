import sys

sys.path.append("./")

import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import os


data_dir = "dataset/ABC/data/compare"
data_list = glob(os.path.join(data_dir, "*.npz"))


def save_ffat_maps(ffat_map, img_path):
    if isinstance(ffat_map, torch.Tensor):
        ffat_map = ffat_map.cpu().numpy()

    rows = 4
    cols = ffat_map.shape[0] // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    for i in range(rows):
        for j in range(cols):
            img = axes[i, j].imshow(ffat_map[i * cols + j])
            fig.colorbar(img, ax=axes[i, j])  # Add color bar for each subplot

    plt.savefig(img_path)


bem_err = []
ours_err = []
neuralsound_err = []
bem_time = []
ours_time = []
neuralsound_time = []

for data_path in tqdm(data_list):
    data = np.load(data_path)

    ours, bem, neuralsound = data["ours"], data["bem"], data["neuralsound"]
    ours_time_, bem_time_, neuralsound_time_ = (
        data["ours_time"],
        data["bem_time"],
        data["neuralsound_time"],
    )
    for i in range(3):
        save_ffat_maps(
            ours[i],
            data_path.replace(".npz", f"_ours_{i}.png").replace("compare", "img"),
        )
        save_ffat_maps(
            bem[i], data_path.replace(".npz", f"_bem_{i}.png").replace("compare", "img")
        )
        save_ffat_maps(
            neuralsound[i],
            data_path.replace(".npz", f"_neuralsound_{i}.png").replace(
                "compare", "img"
            ),
        )

    bem_err.append(0)
    ours_err.append(((ours - bem) ** 2).mean())
    neuralsound_err.append(((neuralsound - bem) ** 2).mean())
    print(ours_err[-1], neuralsound_err[-1])
    bem_time.append(bem_time_)
    ours_time.append(ours_time_)
    neuralsound_time.append(neuralsound_time_)

bem_err = np.array(bem_err)
ours_err = np.array(ours_err)
neuralsound_err = np.array(neuralsound_err)

# analysis and plot
print("bem_err", bem_err.mean(), "+-", bem_err.std())
print("ours_err", ours_err.mean(), "+-", ours_err.std())
print("neuralsound_err", neuralsound_err.mean(), "+-", neuralsound_err.std())

bem_time = np.array(bem_time)
ours_time = np.array(ours_time)
neuralsound_time = np.array(neuralsound_time)

print("bem_time", bem_time.mean(), "+-", bem_time.std())
print("ours_time", ours_time.mean(), "+-", ours_time.std())
print("neuralsound_time", neuralsound_time.mean(), "+-", neuralsound_time.std())
