import numpy as np


def calculate_bin_frequencies(n_fft=128, sampling_rate=16000):
    # For a one-sided spectrogram, there are n_fft//2 + 1 bins
    num_bins = n_fft // 2 + 1
    return [bin_number * sampling_rate / n_fft for bin_number in range(num_bins)]
