import torch
from torch.utils.data import Dataset
import numpy as np
import os
from glob import glob
import meshio
import bempp.api
from tqdm import tqdm
from torch_geometric.data import Data


class TriMesh:
    def __init__(self, vertices, triangles):
        self.vertices = vertices
        self.triangles = triangles

    def normalize(self):
        self.vertices = self.vertices - self.vertices.mean(dim=0, keepdim=True)
        self.vertices = self.vertices / self.vertices.abs().max()

    def random_rotate(self):
        theta = np.random.rand() * 2 * np.pi
        sin, cos = np.sin(theta), np.cos(theta)
        axis = np.random.randint(0, 3)
        if axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        self.vertices = torch.matmul(self.vertices, torch.tensor(matrix).float())

    def random_scale(self):
        # 0.8 - 1.2
        scale = np.random.rand() * 0.4 + 0.8
        self.vertices = self.vertices * scale

    def random_transform(self):
        self.random_rotate()
        self.random_scale()

    def save_obj(self, path):
        with open(path, "w") as f:
            for v in self.vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for t in self.triangles:
                f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")


class MeshDataset(Dataset):
    def __init__(self, mesh_dir, split="train"):
        self.root_dir = mesh_dir
        self.split = split
        self.file_list = glob(mesh_dir + "/*.obj")
        self.file_list.sort()
        if split == "train":
            self.file_list = self.file_list[: int(0.8 * len(self.file_list))]
            self.phase = "train"
        else:
            self.file_list = self.file_list[int(0.8 * len(self.file_list)) :]
            self.phase = "test"

    def __len__(self):
        return len(self.mesh_data_list)

    def pre_process_meshes(self):
        mesh_data_list = []
        print("Pre-processing meshes...")
        for mesh_file in tqdm(self.file_list):
            try:
                mesh = meshio.read(mesh_file)
                vertices = mesh.points
                triangles = mesh.cells_dict["triangle"]
                vertices = torch.from_numpy(vertices).float()
                triangles = torch.from_numpy(triangles).long()
                mesh = TriMesh(vertices, triangles)
                mesh.normalize()
                if triangles.shape[0] < 15000:
                    mesh_data_list.append(mesh)
            except:
                print("Error in loading mesh: ", mesh_file)

        torch.save(mesh_data_list, self.root_dir + f"/{self.phase}_data.pt")

    def load_pre_processed_mesh(self):
        self.mesh_data_list = torch.load(self.root_dir + f"/{self.phase}_data.pt")

    def __getitem__(self, idx):
        mesh = self.mesh_data_list[idx]
        mesh.random_transform()
        vertices = mesh.vertices
        triangles = mesh.triangles
        v1 = vertices[triangles[:, 0]]
        v2 = vertices[triangles[:, 1]]
        v3 = vertices[triangles[:, 2]]
        pos = (v1 + v2 + v3) / 3
        normal = torch.cross(v2 - v1, v3 - v1)
        neumann = torch.randn(len(triangles), 1)
        x = torch.cat([v1 - pos, v2 - pos, v3 - pos, normal, pos, neumann], dim=1)
        vertices_offset = len(vertices)
        return Data(
            x=x,
            pos=pos,
            vertices=vertices,
            triangles=triangles,
            vertices_offset=vertices_offset,
            neumann=neumann,
        )
