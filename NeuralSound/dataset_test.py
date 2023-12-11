import torch
from torch.utils.data import Dataset
import numpy as np
import MinkowskiEngine as ME
import os
import sys
from glob import glob

sys.path.append("..")


def freq2mel(f):
    return np.log10(f / 700 + 1) * 2595


mel_max = freq2mel(20000)
mel_min = freq2mel(20)


def freq2wave_number(f):
    return 2 * np.pi * f / 343


wn_max = freq2wave_number(20000)
wn_min = freq2wave_number(20)


class AcousticDataset(Dataset):
    SUFFIX = ""

    def __init__(self, root_dir, phase):
        self.file_list = glob(f"{root_dir}/*.npz")
        data = np.load(self.file_list[0])
        freqs = data["freqs"]
        freq_num = len(freqs)
        self.mode_num = freq_num

    def __len__(self):
        return len(self.file_list) * self.mode_num

    def __getitem__(self, i):
        index = i // self.mode_num
        mode_idx = i % self.mode_num
        filename = self.file_list[index]
        data = np.load(filename)
        coords, feats_in, feats_out, surface_code, freqs = (
            data["coords"],
            data["feats_in"],
            data["feats_out" + self.SUFFIX],
            data["surface"],
            data["freqs"],
        )
        # print('feats_out' + self.SUFFIX)
        # Random selection
        freq = freqs[mode_idx]
        feats_in = feats_in[:, :, mode_idx]
        feats_out = feats_out[mode_idx][np.newaxis, ...]
        coords = coords - 1
        voxel_num = coords.shape[0]

        # normalize
        coords_feats = (coords / 16 - 1) / 0.5
        feats_in_norm = (feats_in**2).mean() ** 0.5
        feats_in = feats_in / feats_in_norm

        wave_number = freq2wave_number(freq)
        freq_norm = (
            torch.ones(voxel_num, 1) * (wave_number - wn_min) / (wn_max - wn_min)
        )
        voxel_size = 0.15 / 32
        sin_cos_item = torch.tensor(
            [
                np.cos(wave_number * voxel_size / 4),
                np.sin(wave_number * voxel_size / 4),
                np.cos(wave_number * voxel_size / 2),
                np.sin(wave_number * voxel_size / 2),
                np.cos(wave_number * voxel_size / 1),
                np.sin(wave_number * voxel_size / 1),
                np.cos(wave_number * voxel_size * 2),
                np.sin(wave_number * voxel_size * 2),
                np.cos(wave_number * voxel_size * 4),
                np.sin(wave_number * voxel_size * 4),
            ]
        )
        sin_cos = torch.ones(voxel_num, sin_cos_item.shape[-1]) * sin_cos_item

        feats_in = np.concatenate(
            [feats_in, surface_code, freq_norm, sin_cos, coords_feats], axis=1
        )

        # print((feats_out**2).mean()**0.5, (feats_out_r**2).mean()**0.5)
        feats_out = feats_out / feats_in_norm
        feats_out_norm = (feats_out**2).mean() ** 0.5
        # print(feats_out_norm)
        feats_out = feats_out / feats_out_norm
        feats_out_norm = (np.log(feats_out_norm) + 8) / 3

        return (
            coords,
            feats_in,
            feats_out,
            feats_out_norm,
            filename,
        )


class AcousticDatasetFar(AcousticDataset):
    SUFFIX = "_far"


def acoustic_collation_fn(datas):
    (
        coords,
        feats_in,
        feats_out,
        feats_out_norm,
        filename,
    ) = list(zip(*datas))
    # Create batched coordinates for the SparseTensor input
    bcoords = ME.utils.batched_coordinates(coords)
    # Concatenate all lists
    feats_in = torch.from_numpy(np.concatenate(feats_in, axis=0)).float()
    feats_out = torch.from_numpy(np.stack(feats_out, axis=0)).float()
    # freq_norm = torch.tensor(freq_norm).float().unsqueeze(-1)
    feats_out_norm = torch.tensor(feats_out_norm).float().unsqueeze(-1)
    return (
        bcoords,
        feats_in,
        feats_out,
        feats_out_norm,
        filename,
    )
