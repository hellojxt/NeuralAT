from .modalobj.model import StaticObj, get_mesh_center, normalize_vertices
import json
from scipy.spatial.transform import Rotation as R
import torch
from .utils import Visualizer
import os
from src.bem.solver import BEM_Solver
import numpy as np
from glob import glob
import meshio
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def rotate_vertices(vs, rot_axis, rot_degree):
    vs = R.from_euler(rot_axis, rot_degree, degrees=True).apply(vs.cpu().numpy())
    return torch.from_numpy(vs).cuda()


class ObjList:
    def __init__(self, obj_json, data_dir):
        self.vertices_list = []
        self.triangles_list = []
        self.neumann_list = []
        obj_dir = os.path.join(data_dir, obj_json["mesh_dir"])
        print("Loading obj files")
        for obj_path in tqdm(glob(os.path.join(obj_dir, "*_remesh.obj"))):
            obj = StaticObj(obj_path, obj_json["size"])
            self.vertices_list.append(
                torch.tensor(obj.vertices).cuda().to(torch.float32)
            )
            self.triangles_list.append(
                torch.tensor(obj.triangles).cuda().to(torch.int32)
            )
            neumann = torch.zeros(
                len(obj.vertices), dtype=torch.complex64, device="cuda"
            )
            self.neumann_list.append(neumann)
        self.obj_num = len(self.vertices_list)
        self.rot_vec = None
        self.move_vec = None

    def sample(self, rnd):
        idx = int(rnd * self.obj_num)
        self.triangles = self.triangles_list[idx]
        self.vertices = self.vertices_list[idx]
        self.neumann = self.neumann_list[idx]

    def resize(self, factor):
        pass

    def rotation(self, factor):
        pass

    def move(self, factor):
        pass


class ObjAnim:
    def __init__(self, obj_json, data_dir):
        obj = StaticObj(os.path.join(data_dir, obj_json["mesh"]), obj_json["size"])
        self.vertices_base = obj.vertices
        self.triangles = torch.tensor(obj.triangles).cuda().to(torch.int32)
        self.trajectory_points = meshio.read(
            os.path.join(data_dir, obj_json["trajectory"])
        ).points
        self.trajectory_points = normalize_vertices(
            self.trajectory_points, obj_json["trajectory_size"]
        )
        self.trajectory_length = len(self.trajectory_points) - 1
        self.vibration = None if "vibration" not in obj_json else obj_json["vibration"]

        self.neumann = torch.zeros(
            len(self.vertices_base), dtype=torch.complex64, device="cuda"
        )
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
        self.offset = None if "offset" not in obj_json else obj_json["offset"]

    def sample(self, rnd):
        idx = int(rnd * self.trajectory_length)
        x0 = self.trajectory_points[idx]
        x1 = self.trajectory_points[idx + 1]
        v = x1 - x0
        center = (x0 + x1) / 2 + self.offset
        x_axis = np.array([1, 0, 0])
        rotation, _ = R.align_vectors([v], [x_axis])
        self.vertices = rotation.apply(self.vertices_base)
        self.vertices = self.vertices - get_mesh_center(self.vertices) + center
        self.vertices = torch.tensor(self.vertices).cuda().to(torch.float32)

    def resize(self, factor):
        pass

    def rotation(self, factor):
        pass

    def move(self, factor):
        pass


class Obj:
    def __init__(self, obj_json, data_dir):
        obj = StaticObj(os.path.join(data_dir, obj_json["mesh"]), obj_json["size"])
        self.name = obj_json["mesh"]
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
        self.position = (
            None
            if "position" not in obj_json
            else torch.tensor(obj_json["position"]).cuda()
        )
        self.vibration = None if "vibration" not in obj_json else obj_json["vibration"]

        self.neumann = torch.zeros(
            len(self.vertices_base), dtype=torch.complex64, device="cuda"
        )
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
        self.obj_list_num = 0
        self.resize = False
        for obj_json in data["obj"]:
            if "mesh_dir" in obj_json or "trajectory" in obj_json:
                if "mesh_dir" in obj_json:
                    obj = ObjList(obj_json, self.data_dir)
                else:
                    obj = ObjAnim(obj_json, self.data_dir)
                self.obj_list_num += 1
            else:
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
        self.bbox_size = (self.trg_pos_max - self.trg_pos_min).max()
        self.bbox_center = (self.trg_pos_max + self.trg_pos_min) / 2
        self.trg_points = None

    def sample(self, max_resize=2, log=False):
        rot_factors = torch.rand(self.rot_num).cuda()
        move_factors = torch.rand(self.move_num).cuda()
        obj_list_factors = torch.rand(self.obj_list_num).cuda()
        resize_factor = torch.rand(1).cuda()
        freq_factor = torch.rand(1).cuda()
        rot_idx = 0
        move_idx = 0
        obj_list_idx = 0
        for obj in self.objs:
            if isinstance(obj, ObjList):
                obj.sample(obj_list_factors[obj_list_idx].item())
                obj_list_idx += 1
            if isinstance(obj, ObjAnim):
                obj.sample(obj_list_factors[obj_list_idx].item())
                obj_list_idx += 1
            obj.resize(resize_factor.item() * (max_resize - 1) + 1)
            if self.rot_num > 0:
                obj.rotation(rot_factors[rot_idx].item())
                if log and obj.rot_axis is not None:
                    print(obj.name, "rotaion idx:", rot_idx)
                if obj.rot_axis is not None and rot_idx < self.rot_num - 1:
                    rot_idx += 1
            if self.move_num > 0:
                obj.move(move_factors[move_idx].item())
                if log and obj.move_vec is not None:
                    print(obj.name, "move idx:", move_idx)
                if obj.move_vec is not None and move_idx < self.move_num - 1:
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
        self.obj_list_factors = obj_list_factors
        self.resize_factor = resize_factor
        self.freq_factor = freq_factor
        freq_log = (
            self.freq_factor * (self.freq_max_log - self.freq_min_log)
            + self.freq_min_log
        )
        freq = 10**freq_log
        self.k = (2 * np.pi * freq / 343.2).item()

    def solve(self):
        solver = BEM_Solver(self.vertices, self.triangles)
        self.dirichlet = solver.neumann2dirichlet(self.k, self.neumann)

        sample_points_base = torch.rand(
            self.trg_sample_num, 3, device="cuda", dtype=torch.float32
        )
        rs = (sample_points_base[:, 0] + 1) * self.bbox_size
        theta = sample_points_base[:, 1] * 2 * np.pi - np.pi
        phi = sample_points_base[:, 2] * np.pi
        xs = rs * torch.sin(phi) * torch.cos(theta)
        ys = rs * torch.sin(phi) * torch.sin(theta)
        zs = rs * torch.cos(phi)
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.trg_points = torch.stack([xs, ys, zs], dim=-1) + self.bbox_center

        self.trg_factor = sample_points_base
        self.potential = solver.boundary2potential(
            self.k, self.neumann, self.dirichlet, self.trg_points
        ).cpu()

    def show(self):
        vis = Visualizer()
        vis.add_mesh(self.vertices, self.triangles, self.neumann.abs())
        if self.trg_points is not None:
            vis.add_points(self.trg_points, self.potential.abs())
        vis.show()


class EditableModalSound:

    def __init__(self, data_dir, ffat_res=(64, 32), uniform=False):
        with open(f"{data_dir}/config.json", "r") as file:
            js = json.load(file)
            sample_config = js.get("sample", {})
            obj_config = js.get("vibration_obj", {})
            self.size_base = obj_config.get("size")

        data = torch.load(f"{data_dir}/modal_data.pt")
        self.vertices_base = data["vertices"]
        self.triangles = data["triangles"]
        self.neumann_vtx = data["neumann_vtx"]
        self.ks_base = data["ks"]
        self.mode_num = len(self.ks_base)

        self.freq_rate = sample_config.get("freq_rate")
        self.size_rate = sample_config.get("size_rate")
        self.bbox_rate = sample_config.get("bbox_rate")
        self.sample_num = sample_config.get("sample_num")
        self.point_num_per_sample = sample_config.get("point_num_per_sample")

        self.ffat_res = ffat_res
        xs = torch.linspace(0, 1, ffat_res[0], device="cuda", dtype=torch.float32)
        ys = torch.linspace(0, 1, ffat_res[1], device="cuda", dtype=torch.float32)
        self.gridx, self.gridy = torch.meshgrid(xs, ys)
        self.uniform = uniform
        if uniform:
            self.point_num_per_sample = ffat_res[0] * ffat_res[1]

    def sample(self, freqK_base=None, sizeK_base=None):
        if freqK_base is None:
            self.freqK_base = torch.rand(1).cuda()
        else:
            self.freqK_base = freqK_base

        if sizeK_base is None:
            self.sizeK_base = torch.rand(1).cuda()
        else:
            self.sizeK_base = sizeK_base

        self.freqK = self.freqK_base * self.freq_rate
        self.sizeK = 1.0 / (1 + self.sizeK_base * (self.size_rate - 1))

        self.vertices = self.vertices_base * self.sizeK
        self.ks = self.ks_base * self.freqK / self.sizeK**0.5

        if self.uniform:
            sample_points_base = torch.zeros(
                self.ffat_res[0] * self.ffat_res[1],
                3,
                device="cuda",
                dtype=torch.float32,
            )
            sample_points_base[:, 0] = 0.5
            sample_points_base[:, 1] = self.gridx.reshape(-1)
            sample_points_base[:, 2] = self.gridy.reshape(-1)
        else:
            sample_points_base = torch.rand(
                self.point_num_per_sample, 3, device="cuda", dtype=torch.float32
            )

        self.sample_points_base = sample_points_base
        rs = (
            (sample_points_base[:, 0] * (self.bbox_rate - 1) + 1) * self.size_base * 0.7
        )
        theta = sample_points_base[:, 1] * 2 * np.pi - np.pi
        phi = sample_points_base[:, 2] * np.pi
        xs = rs * torch.sin(phi) * torch.cos(theta)
        ys = rs * torch.sin(phi) * torch.sin(theta)
        zs = rs * torch.cos(phi)
        self.trg_points = torch.stack([xs, ys, zs], dim=-1)

        input_x = torch.zeros(
            self.sample_points_base.shape[0],
            3 + 1 + 1,
            dtype=torch.float32,
        )
        input_x[:, :3] = self.sample_points_base
        input_x[:, 3] = self.sizeK_base
        input_x[:, 4] = self.freqK_base
        return input_x

    def solve(self):
        bem_solver = BEM_Solver(self.vertices, self.triangles)
        potentials = []
        for i in range(self.mode_num):
            dirichlet_vtx = bem_solver.neumann2dirichlet(
                self.ks[i].item(), self.neumann_vtx[i]
            )
            potential = bem_solver.boundary2potential(
                self.ks[i].item(), self.neumann_vtx[i], dirichlet_vtx, self.trg_points
            )
            potentials.append(potential)

        self.potentials = torch.stack(potentials, dim=0).cpu()
        return self.potentials

    def show(self, mode_idx=0):
        vis = Visualizer()
        vis.add_mesh(self.vertices, self.triangles, self.neumann_vtx[mode_idx].abs())
        vis.add_points(self.trg_points, self.potentials[mode_idx].abs())
        vis.show()
