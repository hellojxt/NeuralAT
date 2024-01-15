import sys

sys.path.append("./")
from src.timer import Timer
from src.modalsound.model import (
    SNR,
    complex_ssim,
)
import numpy as np
import sys
import matplotlib.pyplot as plt
import configparser
import os
from src.audio import calculate_bin_frequencies


def get_mesh_size(vertices):
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    return (bbox_max - bbox_min).max()


freq_bins = calculate_bin_frequencies()[1:]
freq_bin_num = len(freq_bins)
print(freq_bin_num)
y_list = [i / 10 for i in range(10)]
y_num = len(y_list)

spectrogram = np.zeros((y_num, freq_bin_num, 4))
SNRs = np.zeros((y_num, freq_bin_num, 4))
SSIMs = np.zeros((y_num, freq_bin_num, 4))
cost_times = np.zeros((y_num, freq_bin_num, 4))
root_dir = "dataset/NeuPAT/audio/large_mlp"

for y_i in range(y_num):
    for freq_i in range(freq_bin_num):
        data_dir = f"{root_dir}/{y_list[y_i]:.1f}_{freq_bins[freq_i]:.0f}"
        vertices = np.load(f"{data_dir}/bem.npz")["vertices"]
        mesh_size = get_mesh_size(vertices)
        ffat_map_NeuralSound = np.load(f"{data_dir}/NeuralSound.npz")[
            "ffat_map"
        ].reshape(64, 32)
        cost_time_NeuralSound = np.load(f"{data_dir}/NeuralSound.npz")["cost_time"]
        ffat_map_neuPAT = np.load(f"{data_dir}/neuPAT.npz")["ffat_map"].reshape(64, 32)
        # ys = ((ys + 10e-6) / 10e-6).log10()
        ffat_map_neuPAT = (10**ffat_map_neuPAT) * 10e-6 - 10e-6
        cost_time_neuPAT = np.load(f"{data_dir}/neuPAT.npz")["cost_time"]

        ffat_map_ours = np.load(f"{data_dir}/ours.npz")["ffat_map"].reshape(64, 32)
        cost_time_ours = np.load(f"{data_dir}/ours.npz")["cost_time"]
        ffat_map_bem = np.load(f"{data_dir}/bem.npz")["ffat_map"].reshape(64, 32)
        cost_time_bem = np.load(f"{data_dir}/bem.npz")["cost_time"]
        points = np.load(f"{data_dir}/bem.npz")["points"]
        r = (points**2).sum(-1) ** 0.5
        ffat_map_NeuralSound = (
            ffat_map_NeuralSound / r[0] / 1.225 * (mesh_size / 0.15) ** (5 / 2)
        )
        spectrogram[y_i, freq_i, 0] = np.abs(ffat_map_bem).sum()
        spectrogram[y_i, freq_i, 1] = np.abs(ffat_map_NeuralSound).sum()
        spectrogram[y_i, freq_i, 2] = np.abs(ffat_map_ours).sum()
        spectrogram[y_i, freq_i, 3] = np.abs(ffat_map_neuPAT).sum()
        SNRs[y_i, freq_i, 0] = np.inf
        SNRs[y_i, freq_i, 1] = SNR(ffat_map_NeuralSound, ffat_map_bem)
        SNRs[y_i, freq_i, 2] = SNR(ffat_map_ours, ffat_map_bem)
        SNRs[y_i, freq_i, 3] = SNR(ffat_map_neuPAT, ffat_map_bem)
        SSIMs[y_i, freq_i, 0] = 1.0
        SSIMs[y_i, freq_i, 1] = complex_ssim(ffat_map_NeuralSound, ffat_map_bem)
        SSIMs[y_i, freq_i, 2] = complex_ssim(ffat_map_ours, ffat_map_bem)
        SSIMs[y_i, freq_i, 3] = complex_ssim(ffat_map_neuPAT, ffat_map_bem)
        cost_times[y_i, freq_i, 0] = cost_time_bem
        cost_times[y_i, freq_i, 1] = cost_time_NeuralSound
        cost_times[y_i, freq_i, 2] = cost_time_ours
        cost_times[y_i, freq_i, 3] = cost_time_neuPAT

batch_step = freq_bin_num // 4
for i in range(0, freq_bin_num, batch_step):
    print(
        f"{freq_bins[i]:.0f}Hz-{freq_bins[i + batch_step - 1]:.0f}Hz& Inf & 1.0 &",
        end=" ",
    )
    print(f"{cost_times[:, i : i + batch_step, 0].sum():.0f}s & ", end="")
    print(f"{SNRs[:, i : i + batch_step, 1].mean():.2f} & ", end="")
    print(f"{SSIMs[:, i : i + batch_step, 1].mean():.2f} & ", end="")
    print(f"{cost_times[:, i : i + batch_step, 1].sum():.1f}s & ", end="")
    print(f"{SNRs[:, i : i + batch_step, 2].mean():.2f} & ", end="")
    print(f"{SSIMs[:, i : i + batch_step, 2].mean():.2f} & ", end="")
    print(f"{cost_times[:, i : i + batch_step, 2].sum():.1f}s & ", end="")
    print(f"{SNRs[:, i : i + batch_step, 3].mean():.2f} & ", end="")
    print(f"{SSIMs[:, i : i + batch_step, 3].mean():.2f} & ", end="")
    print(f"{cost_times[:, i : i + batch_step, 3].sum():.2f}s \\\\")


spectrogram = np.log10((spectrogram + 10e-6) / 10e-6)

from matplotlib.font_manager import FontProperties
from matplotlib.image import imread
from matplotlib.gridspec import GridSpec

# Assuming the other parts of your script are already defined (ffat_map_bem, our_maps, snrs, ssims, data_dir)

# Load specific font
font_path = (
    "/home/jxt/.local/share/fonts/LinBiolinum_R.ttf"  # Replace with your font file path
)
font_bold_path = "/home/jxt/.local/share/fonts/2LinBiolinum_RB.ttf"  # Replace with your font file path

my_font = FontProperties(fname=font_path)
my_font_bold = FontProperties(fname=font_bold_path)


vmin, vmax = np.abs(spectrogram[:, :, 0]).min(), np.abs(spectrogram[:, :, 0]).max()
title_pad = 20
font_size = 25

# Define the number of ticks you want to show on each axis
num_y_ticks = 8  # for example
num_x_ticks = 3  # for example
x_start = 0
x_end = 2 * len(y_list)
# Create figure
fig = plt.figure(figsize=(8, 8))
gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1])

for i in range(4):
    ax = plt.subplot(gs[i])
    ax.imshow(
        np.abs(spectrogram[:, :, i].T),
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=[x_start, x_end, freq_bins[0], freq_bins[-1]],
        aspect="auto",
    )  # Stretching x-axis with extent and interpolation

    # Set title for each subplot
    title = ["BEM", "NeuralSound", "MCAT", "MCAT+NC"][i]
    ax.set_title(title, fontproperties=my_font, fontsize=font_size, pad=title_pad)

    # Set ticks
    y_ticks = np.linspace(freq_bins[0], freq_bins[-1], num_y_ticks)
    ax.set_yticks(y_ticks)
    x_ticks = np.linspace(x_start, x_end, num_x_ticks)
    ax.set_xticks(x_ticks)

    # Set tick labels for the y-axis
    if i == 0:
        ax.set_yticklabels(
            np.round(np.linspace(freq_bins[0], freq_bins[-1], num_y_ticks), 2)
        )
        ax.set_ylabel("Frequency (Hz)", fontproperties=my_font, fontsize=font_size / 2)
    else:
        # Hide y-axis labels and ticks for the 2nd to 4th subplots
        ax.set_yticklabels([])
        ax.set_ylabel("")

    # Set x-axis label
    ax.set_xlabel(
        "Phone's Y-axis Position", fontproperties=my_font, fontsize=font_size / 2
    )
    # Rotate y-axis tick labels to horizontal
    ax.tick_params(axis="y", labelrotation=0)

plt.tight_layout()
plt.savefig(f"{root_dir}/spectrogram.png")
