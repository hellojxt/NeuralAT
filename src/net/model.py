import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy as np

src_pos_config = {
    "otype": "DenseGrid",
    "n_dims_to_encode": 3,
    "n_levels": 8,
    "n_features_per_level": 4,
    "base_resolution": 2,
    "per_level_scale": 2.0,
    "interpolation": "Smoothstep",
}

trg_pos_config = {
    "otype": "DenseGrid",
    "n_dims_to_encode": 3,
    "n_levels": 8,
    "n_features_per_level": 4,
    "base_resolution": 2,
    "per_level_scale": 2.0,
    "interpolation": "Smoothstep",
}

src_rot_config = {
    "otype": "DenseGrid",
    "n_dims_to_encode": 4,
    "n_levels": 4,
    "n_features_per_level": 4,
    "base_resolution": 2,
    "per_level_scale": 2.0,
    "interpolation": "Smoothstep",
}

freq_config = {"otype": "Frequency", "n_frequencies": 12}

network_config = {
    "otype": "CutlassMLP",
    # "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 128,
    "n_hidden_layers": 4,
}


class NeuPAT(nn.Module):
    def __init__(self, src_move=False, trg_move=False, src_rot=False, freq_num=0):
        super().__init__()

        var_config = {
            "otype": config["grid_type"],
            "n_dims_to_encode": 2,
            "n_levels": config["n_levels"],
            "n_features_per_level": config["n_features_per_level"],
            "base_resolution": config["base_resolution"],
            "per_level_scale": config["per_level_scale"],
            "interpolation": "Smoothstep",
        }

        for _ in range(var_num * 3):
            encoding_config["nested"].append(var_config)

        encoding_config["nested"].append(direction_normal_config)
        encoding_config["nested"].append(direction_normal_config)
        encoding_config["nested"].append(reflectance_config)

        network_config["n_neurons"] = config["n_neurons"]
        network_config["n_hidden_layers"] = config["n_hidden_layers"]

        n_input_dims = 3 + 3 * var_num * 2 + 3 + 3 + 3
        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims, n_output_dims, encoding_config, network_config
        )

    def forward(self, pos, dirs, normal, albedo, vars: np.ndarray):
        with dr.suspend_grad():
            pos = ((pos - self.bb_min) / (self.bb_max - self.bb_min)).torch()
            wi = dirs.torch()
            n = normal.torch()
            f_d = albedo.torch()
            vars = torch.from_numpy(vars)

            # there are some nan values due to scene.ray_intersect()
            wi[wi.isnan()] = 0.0
            n[n.isnan()] = 0.0

        x = pos.clone()
        for i in range(vars.shape[0]):
            v = torch.ones_like(x[:, [0]]) * vars[i]
            x = torch.cat([x, pos[:, [0]], v, pos[:, [1]], v, pos[:, [2]], v], dim=1)
        x = torch.cat([x, wi, n, f_d], dim=1)

        # assert(x.isnan().sum() == 0)

        x = self.model(x).to(torch.float32).abs()
        if self.factorize_reflectance:
            if self.n_output_dims == 3:
                x *= f_d
            elif self.n_output_dims == 6:
                x[:, :3] *= f_d
                x[:, 3:] *= f_d
            else:
                raise NotImplementedError
        return x


class DNRFieldRough(nn.Module):
    needsRoughness = True

    def __init__(self, bbox, config, var_num, n_output_dims=3):
        super().__init__()
        self.bb_min = bbox.min
        self.bb_max = bbox.max
        self.factorize_reflectance = config["factorize_reflectance"]
        self.n_output_dims = n_output_dims

        encoding_config = {"otype": "Composite", "nested": []}
        encoding_config["nested"].append(grid_config)

        var_config = {
            "otype": config["grid_type"],
            "n_dims_to_encode": 2,
            "n_levels": config["n_levels"],
            "n_features_per_level": config["n_features_per_level"],
            "base_resolution": config["base_resolution"],
            "per_level_scale": config["per_level_scale"],
            "interpolation": "Smoothstep",
        }

        for _ in range(var_num * 3):
            encoding_config["nested"].append(var_config)

        encoding_config["nested"].append(direction_normal_config)
        encoding_config["nested"].append(direction_normal_config)
        encoding_config["nested"].append(reflectance_config)
        encoding_config["nested"].append(roughness_config)

        network_config["n_neurons"] = config["n_neurons"]
        network_config["n_hidden_layers"] = config["n_hidden_layers"]

        n_input_dims = 3 + 3 * var_num * 2 + 3 + 3 + 3 + 1
        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims, n_output_dims, encoding_config, network_config
        )

    def forward(self, pos, dirs, normal, albedo, roughness, vars: np.ndarray):
        with dr.suspend_grad():
            pos = ((pos - self.bb_min) / (self.bb_max - self.bb_min)).torch()
            wi = dirs.torch()
            n = normal.torch()
            f_d = albedo.torch()
            r = roughness.torch()[:, None]
            vars = torch.from_numpy(vars)

            # there are some nan values due to scene.ray_intersect()
            wi[wi.isnan()] = 0.0
            n[n.isnan()] = 0.0

        x = pos.clone()
        for i in range(vars.shape[0]):
            v = torch.ones_like(x[:, [0]]) * vars[i]
            x = torch.cat([x, pos[:, [0]], v, pos[:, [1]], v, pos[:, [2]], v], dim=1)
        x = torch.cat([x, wi, n, f_d, r], dim=1)

        # assert(x.isnan().sum() == 0)

        x = self.model(x).to(torch.float32).abs()
        if self.factorize_reflectance:
            if self.n_output_dims == 3:
                x *= f_d
            else:
                raise NotImplementedError
        return x


class DNRFieldFull(nn.Module):
    def __init__(self, bbox, config, var_num, n_output_dims=3):
        super().__init__()
        self.bb_min = bbox.min
        self.bb_max = bbox.max

        encoding_config = {"otype": "Composite", "nested": []}
        encoding_config["nested"].append(grid_config)

        var_config = {
            "otype": config["grid_type"],
            "n_dims_to_encode": 2,
            "n_levels": config["n_levels"],
            "n_features_per_level": config["n_features_per_level"],
            "base_resolution": config["base_resolution"],
            "per_level_scale": config["per_level_scale"],
            "interpolation": "Smoothstep",
        }

        for _ in range(var_num * (3 + var_num)):
            encoding_config["nested"].append(var_config)

        encoding_config["nested"].append(direction_normal_config)
        encoding_config["nested"].append(direction_normal_config)
        encoding_config["nested"].append(reflectance_config)

        network_config["n_neurons"] = config["n_neurons"]
        network_config["n_hidden_layers"] = config["n_hidden_layers"]

        n_input_dims = 3 + (3 + var_num) * var_num * 2 + 3 + 3 + 3
        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims, n_output_dims, encoding_config, network_config
        )

    def forward(self, pos, dirs, normal, albedo, vars: np.ndarray):
        with dr.suspend_grad():
            pos = ((pos - self.bb_min) / (self.bb_max - self.bb_min)).torch()
            wi = dirs.torch()
            n = normal.torch()
            f_d = albedo.torch()
            vars = torch.from_numpy(vars)

            # there are some nan values due to scene.ray_intersect()
            wi[wi.isnan()] = 0.0
            n[n.isnan()] = 0.0

        x = pos.clone()

        for i in range(vars.shape[0]):
            v = torch.ones_like(x[:, [0]]) * vars[i]
            x = torch.cat([x, pos[:, [0]], v, pos[:, [1]], v, pos[:, [2]], v], dim=1)

        for i in range(vars.shape[0]):
            vi = torch.ones_like(x[:, [0]]) * vars[i]
            for j in range(vars.shape[0]):
                vj = torch.ones_like(x[:, [0]]) * vars[j]
                x = torch.cat([x, vi, vj], dim=1)

        x = torch.cat([x, wi, n, f_d], dim=1)

        # assert(x.isnan().sum() == 0)

        x = self.model(x).to(torch.float32).abs()
        if self.factorize_reflectance:
            if self.n_output_dims == 3:
                x *= f_d
            elif self.n_output_dims == 6:
                x[:, :3] *= f_d
                x[:, 3:] *= f_d
            else:
                raise NotImplementedError
        return x
