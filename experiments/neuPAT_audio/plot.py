import sys

sys.path.append("./")
from src.timer import Timer
from src.modalobj.model import (
    SNR,
    complex_ssim,
)
import numpy as np
import sys
import matplotlib.pyplot as plt
import configparser
import os


def get_mesh_size(vertices):
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    return (bbox_max - bbox_min).max()


data_dir = sys.argv[1]
vertices = np.load(f"{data_dir}/bem.npz")["vertices"]
mesh_size = get_mesh_size(vertices)
print(mesh_size)
ffat_map_NeuralSound = np.load(f"{data_dir}/NeuralSound.npz")["ffat_map"].reshape(
    -1, 64, 32
)
cost_time_NeuralSound = np.load(f"{data_dir}/NeuralSound.npz")["cost_time"]
ffat_map_neuPAT = np.load(f"{data_dir}/neuPAT.npz")["ffat_map"].reshape(-1, 64, 32)
# ys = ((ys + 10e-6) / 10e-6).log10()
ffat_map_neuPAT = (10**ffat_map_neuPAT) * 10e-6 - 10e-6
cost_time_neuPAT = np.load(f"{data_dir}/neuPAT.npz")["cost_time"]

ffat_map_ours = np.load(f"{data_dir}/ours.npz")["ffat_map"].reshape(-1, 64, 32)
cost_time_ours = np.load(f"{data_dir}/ours.npz")["cost_time"]
ffat_map_bem = np.load(f"{data_dir}/bem.npz")["ffat_map"].reshape(-1, 64, 32)
cost_time_bem = np.load(f"{data_dir}/bem.npz")["cost_time"]
points = np.load(f"{data_dir}/bem.npz")["points"]
r = (points**2).sum(-1) ** 0.5
ffat_map_NeuralSound = (
    ffat_map_NeuralSound / r[0] / 1.225 * (mesh_size / 0.15) ** (5 / 2)
)

index = 0
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

# Set color scale limits
vmin, vmax = np.abs(ffat_map_bem[index]).min(), np.abs(ffat_map_bem[index]).max()
title_pad = 20
font_size = 25
# Create figure
fig = plt.figure(figsize=(28, 7))
gs = GridSpec(1, 10, width_ratios=[2, 1, 1, 1, 1, 0.3, 1, 1, 1, 1])

# mesh_render_img = imread(f"{data_dir}/mesh_render.png")
# left_index = mesh_render_img.shape[1] // 6
# right_index = -left_index if left_index != 0 else mesh_render_img.shape[1]
# cropped_img = mesh_render_img[:, left_index:right_index]

# ax = plt.subplot(gs[0])
# ax.imshow(cropped_img)
# ax.text(
#     0.5,
#     0.09,
#     "SNR | SSIM",
#     transform=ax.transAxes,
#     ha="center",
#     fontproperties=my_font,
#     fontsize=font_size,
# )
# if len(sys.argv) > 2:
#     ax.text(
#         0.5,
#         0.88,
#         "Mesh",
#         transform=ax.transAxes,
#         ha="center",
#         fontproperties=my_font,
#         fontsize=font_size,
#     )
# ax.axis("off")

# Plot the first image
ax = plt.subplot(gs[1])

ax.imshow(
    np.abs(ffat_map_bem[index]).reshape(64, 32), cmap="viridis", vmin=vmin, vmax=vmax
)
ax.text(
    16,
    70,
    f"Inf | 1.0",
    ha="center",
    fontproperties=my_font,
    fontsize=font_size,
)

if len(sys.argv) > 2:
    plt.title("BEM", fontproperties=my_font, fontsize=font_size, pad=title_pad)
plt.axis("off")

# print(f"BEM: Inf | 1.0 | {cost_time_bem:.3f}")

ax = plt.subplot(gs[2])
ax.imshow(ffat_map_NeuralSound[index], cmap="viridis", vmin=vmin, vmax=vmax)
ax.text(
    16,
    70,
    f"{SNR(ffat_map_NeuralSound[index], ffat_map_bem[index]):.2f} | {complex_ssim(ffat_map_NeuralSound[index], ffat_map_bem[index]):.2f}",
    ha="center",
    fontproperties=my_font,
    fontsize=font_size,
)
if len(sys.argv) > 2:
    plt.title("NeuralSound", fontproperties=my_font, fontsize=font_size, pad=title_pad)
plt.axis("off")

# SNRs = []
# SSIMs = []
# for i in range(len(ffat_map_NeuralSound)):
#     SNRs.append(SNR(ffat_map_NeuralSound[i], ffat_map_bem[i]))
#     SSIMs.append(complex_ssim(ffat_map_NeuralSound[i], ffat_map_bem[i]))
# print(
#     f"NeuralSound: {np.mean(SNRs):.2f} | {np.mean(SSIMs):.2f} | {cost_time_NeuralSound:.3f}"
# )

# Plot other images and add titles
ax = plt.subplot(gs[3])
ax.imshow(
    np.abs(ffat_map_ours[index]).reshape(64, 32), cmap="viridis", vmin=vmin, vmax=vmax
)
ax.text(
    16,
    70,
    f"{SNR(ffat_map_ours[index], ffat_map_bem[index]):.2f} | {complex_ssim(ffat_map_ours[index], ffat_map_bem[index]):.2f}",
    ha="center",
    fontproperties=my_font,
    fontsize=font_size,
)
if len(sys.argv) > 2:
    plt.title("MCAT", fontproperties=my_font, fontsize=font_size, pad=title_pad)
plt.axis("off")

# SNRs = []
# SSIMs = []
# for i in range(len(ffat_map_ours)):
#     SNRs.append(SNR(ffat_map_ours[i], ffat_map_bem[i]))
#     SSIMs.append(complex_ssim(ffat_map_ours[i], ffat_map_bem[i]))
# print(f"Ours: {np.mean(SNRs):.2f} | {np.mean(SSIMs):.2f} | {cost_time_ours:.3f}")

ax = plt.subplot(gs[4])
ax.imshow(
    np.abs(ffat_map_neuPAT[index]).reshape(64, 32), cmap="viridis", vmin=vmin, vmax=vmax
)

ax.text(
    16,
    70,
    f"{SNR(ffat_map_neuPAT[index], ffat_map_bem[index]):.2f} | {complex_ssim(ffat_map_neuPAT[index], ffat_map_bem[index]):.2f}",
    ha="center",
    fontproperties=my_font,
    fontsize=font_size,
)
if len(sys.argv) > 2:
    plt.title("MCAT+NC", fontproperties=my_font, fontsize=font_size, pad=title_pad)
plt.axis("off")

# SNRs = []
# SSIMs = []
# for i in range(len(ffat_map_neuPAT)):
#     SNRs.append(SNR(ffat_map_neuPAT[i], ffat_map_bem[i]))
#     SSIMs.append(complex_ssim(ffat_map_neuPAT[i], ffat_map_bem[i]))

# print(f"neuPAT: {np.mean(SNRs):.2f} | {np.mean(SSIMs):.2f} | {cost_time_neuPAT:.3f}")

# index = ind2
# vmin, vmax = np.abs(ffat_map_bem[index]).min(), np.abs(ffat_map_bem[index]).max()
# ax = plt.subplot(gs[6])
# ax.imshow(
#     np.abs(ffat_map_bem[index]).reshape(64, 32), cmap="viridis", vmin=vmin, vmax=vmax
# )
# ax.text(
#     16,
#     70,
#     f"Inf | 1.0",
#     ha="center",
#     fontproperties=my_font,
#     fontsize=font_size,
# )

# if len(sys.argv) > 2:
#     plt.title("BEM", fontproperties=my_font, fontsize=font_size, pad=title_pad)

# plt.axis("off")

# ax = plt.subplot(gs[7])

# ax.imshow(ffat_map_NeuralSound[index], cmap="viridis", vmin=vmin, vmax=vmax)
# ax.text(
#     16,
#     70,
#     f"{SNR(ffat_map_NeuralSound[index], ffat_map_bem[index]):.2f} | {complex_ssim(ffat_map_NeuralSound[index], ffat_map_bem[index]):.2f}",
#     ha="center",
#     fontproperties=my_font,
#     fontsize=font_size,
# )
# if len(sys.argv) > 2:
#     plt.title("NeuralSound", fontproperties=my_font, fontsize=font_size, pad=title_pad)
# plt.axis("off")

# # Plot other images and add titles
# ax = plt.subplot(gs[8])
# ax.imshow(
#     np.abs(ffat_map_ours[index]).reshape(64, 32), cmap="viridis", vmin=vmin, vmax=vmax
# )
# ax.text(
#     16,
#     70,
#     f"{SNR(ffat_map_ours[index], ffat_map_bem[index]):.2f} | {complex_ssim(ffat_map_ours[index], ffat_map_bem[index]):.2f}",
#     ha="center",
#     fontproperties=my_font,
#     fontsize=font_size,
# )
# if len(sys.argv) > 2:
#     plt.title("Ours", fontproperties=my_font, fontsize=font_size, pad=title_pad)
# plt.axis("off")

# # Add text below the image

# ax = plt.subplot(gs[9])
# ax.imshow(
#     np.abs(ffat_map_neuPAT[index]).reshape(64, 32), cmap="viridis", vmin=vmin, vmax=vmax
# )

# ax.text(
#     16,
#     70,
#     f"{SNR(ffat_map_neuPAT[index], ffat_map_bem[index]):.2f} | {complex_ssim(ffat_map_neuPAT[index], ffat_map_bem[index]):.2f}",
#     ha="center",
#     fontproperties=my_font,
#     fontsize=font_size,
# )

# if len(sys.argv) > 2:
#     plt.title("neuPAT", fontproperties=my_font, fontsize=font_size, pad=title_pad)

# plt.axis("off")


plt.savefig(f"{data_dir}/scale_comparison.png")
