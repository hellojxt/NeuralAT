import numpy as np


def calculate_bin_frequencies(n_fft=128, sampling_rate=16000):
    # For a one-sided spectrogram, there are n_fft//2 + 1 bins
    num_bins = n_fft // 2 + 1
    return [bin_number * sampling_rate / n_fft for bin_number in range(num_bins)]


import matplotlib.pyplot as plt

# plt.imshow(nc_spec)
# plt.show()

from scipy.io import wavfile
import librosa
import torchaudio
import torch
from scipy.interpolate import RegularGridInterpolator


def apply_spec_mask_to_audio(
    audio_id,
    mask_spec,
    frame_num,
    animation_frame_rate=30,
    trg_sample_rate=16000,
    n_fft=256,
):
    sample_rate, data = wavfile.read(f"dataset/audio/{audio_id}.wav")
    data = data[:, 0]
    data = (data / 32768).astype(np.float32)
    data = librosa.resample(data, orig_sr=sample_rate, target_sr=trg_sample_rate)
    sample_rate = trg_sample_rate
    require_audio_frame = int(frame_num / animation_frame_rate * sample_rate)
    if len(data) < require_audio_frame:
        data_repeat_num = require_audio_frame // len(data) + 1
        data = np.tile(data, data_repeat_num)
    data = data[: int(frame_num / animation_frame_rate * sample_rate)]
    torch_data = torch.from_numpy(data)
    spec_audio = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=n_fft // 2, power=None
    )(torch_data.unsqueeze(0)).squeeze(0)

    def plot(spec, idx, title):
        def plot_spec(ax, spec, title):
            ax.set_title(title)
            ax.imshow(librosa.amplitude_to_db(spec), origin="lower", aspect="auto")
            plt.axis("off")

        ax = plt.subplot(3, 1, idx)
        plot_spec(ax, torch.abs(spec), title="")

    plot(spec_audio, 1, "original")

    spec_audio = spec_audio.cpu().numpy()
    x = np.linspace(0, 1, mask_spec.shape[1])
    y = np.linspace(0, 1, mask_spec.shape[0])
    new_x = np.linspace(0, 1, spec_audio.shape[1])
    new_y = np.linspace(0, 1, spec_audio.shape[0])
    interp_function = RegularGridInterpolator((y, x), mask_spec)
    new_y, new_x = np.meshgrid(new_y, new_x, indexing="ij")
    new_points = np.array([new_y.ravel(), new_x.ravel()]).T
    interped_nc_spec = interp_function(new_points).reshape(*spec_audio.shape)
    spec_audio = spec_audio * interped_nc_spec
    spec_audio = torch.from_numpy(spec_audio)
    plot(torch.from_numpy(interped_nc_spec), 2, "mask")
    plot(spec_audio, 3, "masked")
    plt.show()

    audio = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft)(
        spec_audio.unsqueeze(0)
    ).squeeze(0)
    audio = audio.numpy()
    audio = audio / np.abs(audio).max() * 0.9
    audio = (audio * 32768).astype(np.int16)
    return audio
