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
    "n_hidden_layers": 2,
}


class ComplexMLPS(torch.nn.Module):
    def __init__(self, use_tcnn=True):
        super().__init__()
        out_channel = 2
        n_dims_to_encode = 0
        for cfg in encoding_cfg["nested"]:
            n_dims_to_encode += cfg["n_dims_to_encode"]
        if use_tcnn:
            self.net = tcnn.NetworkWithInputEncoding(
                n_dims_to_encode, out_channel, encoding_cfg, network_cfg
            )
        else:
            encoding = tcnn.Encoding(
                n_dims_to_encode, encoding_cfg, dtype=torch.float32
            )
            layers = [torch.nn.Linear(encoding.n_output_dims, network_cfg["n_neurons"])]
            for _ in range(network_cfg["n_hidden_layers"]):
                layers.append(torch.nn.ReLU())
                layers.append(
                    torch.nn.Linear(network_cfg["n_neurons"], network_cfg["n_neurons"])
                )
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(network_cfg["n_neurons"], out_channel))
            self.net = torch.nn.Sequential(encoding, *layers).cuda()

    def forward(self, x):
        y = self.net(x)
        return torch.view_as_complex(y.float())


def get_mlps(out_channel, use_tcnn=True):
    n_dims_to_encode = 0
    for cfg in encoding_cfg["nested"]:
        n_dims_to_encode += cfg["n_dims_to_encode"]
    if use_tcnn:
        return tcnn.NetworkWithInputEncoding(
            n_dims_to_encode, out_channel, encoding_cfg, network_cfg
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
        layers.append(torch.nn.Linear(network_cfg["n_neurons"], out_channel))
        return torch.nn.Sequential(encoding, *layers).cuda()
