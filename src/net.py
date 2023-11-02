import torch
import commentjson as json
import tinycudann as tcnn

encoding_cfg = {
    "otype": "Composite",
    "nested": [
        {
            "n_dims_to_encode": 3,
            "otype": "Grid",
            "n_levels": 8,
            "n_features_per_level": 2,
            "base_resolution": 16,
            "per_level_scale": 2.0,
        },
        {"n_dims_to_encode": 3, "otype": "SphericalHarmonics", "degree": 4},
    ],
}

network_cfg = {
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 128,
    "n_hidden_layers": 3,
}


def get_mlps(use_tcnn=True):
    n_dims_to_encode = 0
    for cfg in encoding_cfg["nested"]:
        n_dims_to_encode += cfg["n_dims_to_encode"]
    if use_tcnn:
        return tcnn.NetworkWithInputEncoding(
            n_dims_to_encode, 1, encoding_cfg, network_cfg
        )
    else:
        encoding = tcnn.Encoding(n_dims_to_encode, encoding_cfg, dtype=torch.float32)
        layers = [torch.nn.Linear(encoding.n_output_dims, network_cfg["n_neurons"])]
        for _ in range(network_cfg["n_hidden_layers"]):
            layers.append(torch.nn.ReLU())
            layers.append(
                torch.nn.Linear(network_cfg["n_neurons"], network_cfg["n_neurons"])
            )
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(network_cfg["n_neurons"], 1))
        return torch.nn.Sequential(encoding, *layers).cuda()
