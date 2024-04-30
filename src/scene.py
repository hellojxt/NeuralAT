from .modalobj.model import StaticObj
import json
import scipy.spatial.transform as transform
import torch
from .utils import Visualizer
import os
from src.bem.solver import BEM_Solver
import numpy as np


def rotate_vertices(vs, rot_axis, rot_degree):
    vs = transform.Rotation.from_euler(rot_axis, rot_degree, degrees=True).apply(
        vs.cpu().numpy()
    )
    return torch.from_numpy(vs).cuda()


class Obj:
    def __init__(self, obj_json, data_dir):
        obj = StaticObj(os.path.join(data_dir, obj_json["mesh"]), obj_json["size"])
        self.vertices_base = torch.tensor(obj.vertices).cuda().to(torch.float32)
        self.triangles = torch.tensor(obj.triangles).cuda().to(torch.int32)
        self.resize_base = (
            torch.zeros(3).cuda()
            if "resize" not in obj_json
            else torch.tensor(obj_json["resize"]).cuda()
        )
        self.rot_axis = None if "rot_axis" not in obj_json else obj_json["rot_axis"]
        self.rot_pos = (
            None
            if "rot_pos" not in obj_json
            else torch.tensor(obj_json["rot_pos"]).cuda()
        )
        self.rot_max_deg = (
            None if "rot_max_deg" not in obj_json else obj_json["rot_max_deg"]
        )
        self.move_vec = (
            None if "move" not in obj_json else torch.tensor(obj_json["move"]).cuda()
        )
        self.position = torch.tensor(obj_json["position"]).cuda()
        self.vibration = None if "vibration" not in obj_json else obj_json["vibration"]

        self.neumann = torch.zeros(
            len(self.vertices_base), dtype=torch.complex64
        ).cuda()
        if self.vibration is not None:
            self.neg = False
            if "-" in self.vibration:
                self.neg = True
                self.vibration = self.vibration.replace("-", "")
            idx = 0 if "x" in self.vibration else 1 if "y" in self.vibration else 2
            if self.neg:
                self.neumann[self.vertices_base[:, idx] < 0] = 1
            else:
                self.neumann[self.vertices_base[:, idx] > 0] = 1

    def resize(self, factor):
        self.vertices = self.vertices_base.clone()
        self.resize_vec = self.resize_base * factor
        self.resize_vec[self.resize_vec == 0] = 1
        self.vertices *= self.resize_vec

    def rotation(self, factor):
        if self.rot_axis is not None:
            self.vertices = (
                rotate_vertices(
                    self.vertices - self.rot_pos * self.resize_vec,
                    self.rot_axis,
                    factor * self.rot_max_deg,
                )
                + self.rot_pos * self.resize_vec
            )

    def move(self, factor):
        if self.move_vec is not None:
            self.vertices = self.vertices + self.move_vec * factor
        self.vertices += self.position


class Scene:
    def __init__(self, json_path):
        with open(json_path, "r") as file:
            data = json.load(file)
        self.objs = []
        self.data_dir = os.path.dirname(json_path)
        self.rot_num = 0
        self.move_num = 0
        self.resize = False
        for obj_json in data["obj"]:
            obj = Obj(obj_json, self.data_dir)
            if obj.rot_axis is not None:
                self.rot_num += 1
            if obj.move_vec is not None:
                self.move_num += 1
            if obj.resize_base is not None:
                self.resize = True
            self.objs.append(obj)
        solver_json = data["solver"]
        self.src_sample_num = solver_json["src_sample_num"]
        self.trg_sample_num = solver_json["trg_sample_num"]
        self.freq_min = solver_json["freq_min"]
        self.freq_max = solver_json["freq_max"]
        self.freq_min_log = np.log10(self.freq_min)
        self.freq_max_log = np.log10(self.freq_max)
        self.trg_pos_min = torch.tensor(solver_json["trg_pos_min"]).cuda()
        self.trg_pos_max = torch.tensor(solver_json["trg_pos_max"]).cuda()
        self.trg_points = None

    def sample(self, max_resize=2):
        rot_factors = torch.rand(self.rot_num).cuda()
        move_factors = torch.rand(self.move_num).cuda()
        resize_factor = torch.rand(1).cuda()
        freq_factor = torch.rand(1).cuda()
        rot_idx = 0
        move_idx = 0
        for obj in self.objs:
            obj.resize(resize_factor.item() * (max_resize - 1) + 1)
            obj.rotation(rot_factors[rot_idx].item())
            if obj.rot_axis is not None:
                rot_idx += 1
            obj.move(move_factors[move_idx].item())
            if obj.move_vec is not None:
                move_idx += 1
        self.vertices = torch.zeros(0, 3).cuda().to(torch.float32)
        self.triangles = torch.zeros(0, 3).cuda().to(torch.int32)
        self.neumann = torch.zeros(0).cuda().to(torch.complex64)
        for obj in self.objs:
            self.triangles = torch.cat(
                [self.triangles, obj.triangles + len(self.vertices)]
            )
            self.vertices = torch.cat([self.vertices, obj.vertices])
            self.neumann = torch.cat([self.neumann, obj.neumann])
        self.vertices = self.vertices.contiguous().float()
        self.triangles = self.triangles.contiguous().int()
        self.rot_factors = rot_factors
        self.move_factors = move_factors
        self.resize_factor = resize_factor
        self.freq_factor = freq_factor

    def solve(self):
        solver = BEM_Solver(self.vertices, self.triangles)
        freq_log = (
            self.freq_factor * (self.freq_max_log - self.freq_min_log)
            + self.freq_min_log
        )
        freq = 10**freq_log
        k = (2 * np.pi * freq / 343.2).item()
        self.dirichlet = solver.neumann2dirichlet(k, self.neumann)
        sample_points_base = torch.rand(
            self.trg_sample_num, 3, device="cuda", dtype=torch.float32
        )
        self.trg_factor = sample_points_base
        self.trg_points = (
            sample_points_base * (self.trg_pos_max - self.trg_pos_min)
            + self.trg_pos_min
        )
        self.potential = (
            solver.boundary2potential(k, self.neumann, self.dirichlet, self.trg_points)
            .abs()
            .cpu()
        )

    def show(self):
        vis = Visualizer()
        vis.add_mesh(self.vertices, self.triangles, self.neumann.abs())
        if self.trg_points is not None:
            vis.add_points(self.trg_points, self.potential)
        vis.show()
