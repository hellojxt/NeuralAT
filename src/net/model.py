import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy as np

src_pos_config = {
    "otype": "HashGrid",
    "n_dims_to_encode": 3,
    "n_levels": 8,
    "n_features_per_level": 4,
    "base_resolution": 8,
    "per_level_scale": 2.0,
}

trg_pos_config = {
    "otype": "HashGrid",
    "n_dims_to_encode": 3,
    "n_levels": 8,
    "n_features_per_level": 4,
    "base_resolution": 8,
    "per_level_scale": 2.0,
}

# trg_pos_config = {"otype": "SphericalHarmonics", "n_dims_to_encode": 3, "degree": 4}

src_rot_config = {
    "otype": "HashGrid",
    "n_dims_to_encode": 4,
    "n_levels": 4,
    "n_features_per_level": 4,
    "base_resolution": 8,
    "per_level_scale": 2.0,
}

freq_config = {"otype": "Frequency", "n_dims_to_encode": 4, "n_frequencies": 12}

network_config = {
    "otype": "CutlassMLP",
    # "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 128,
    "n_hidden_layers": 4,
}


class NeuPAT(nn.Module):
    def __init__(
        self,
        n_output_dims,
        src_move=False,
        trg_move=False,
        src_rot=False,
        freq_num=0,
        n_neurons=128,
        n_hidden_layers=4,
    ):
        super().__init__()
        encoding_config = {"otype": "Composite", "nested": []}
        n_input_dims = 0
        if src_move:
            encoding_config["nested"].append(src_pos_config)
            n_input_dims += 3
        if trg_move:
            encoding_config["nested"].append(trg_pos_config)
            n_input_dims += 3
        if src_rot:
            encoding_config["nested"].append(src_rot_config)
            n_input_dims += 4
        if freq_num > 0:
            freq_config["n_dims_to_encode"] = freq_num
            encoding_config["nested"].append(freq_config)
            n_input_dims += freq_num

        network_config["n_neurons"] = n_neurons
        network_config["n_hidden_layers"] = n_hidden_layers

        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims, n_output_dims, encoding_config, network_config
        )

    def forward(self, x):
        x = self.model(x).to(torch.float32).abs()
        return x
