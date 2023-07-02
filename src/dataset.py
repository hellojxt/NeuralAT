import torch
from torch.utils.data import Dataset
import numpy as np
import os
from glob import glob
import meshio
import bempp.api
from tqdm import tqdm


class MeshData:
    def __init__(self, vertices=None, triangles=None):
        self.vertices = vertices
        self.triangles = triangles
        if (vertices is not None) and (triangles is not None):
            self.adj = self.get_adj()

    def to_torch(self, device="cuda"):
        self.vertices = torch.from_numpy(self.vertices).float().to(device)
        self.triangles = torch.from_numpy(self.triangles).long().to(device)
        self.adj = torch.from_numpy(self.adj).long().to(device)

    def get_adj(self):
        grid = bempp.api.Grid(self.vertices.T, self.triangles.T)
        adj = grid.edge_adjacency.T[:, :2]
        return adj

    def save(self, filename):
        np.savez(
            filename, vertices=self.vertices, triangles=self.triangles, adj=self.adj
        )

    def load(self, filename):
        data = np.load(filename)
        self.vertices = data["vertices"]
        self.triangles = data["triangles"]
        self.adj = data["adj"]


class MeshDataset(Dataset):
    def __init__(self, mesh_dir, split="train"):
        self.root_dir = mesh_dir
        self.split = split
        self.file_list = glob(mesh_dir + "/*.obj")
        self.file_list.sort()
        if split == "train":
            self.file_list = self.file_list[: int(0.8 * len(self.file_list))]
        elif split == "val":
            self.file_list = self.file_list[int(0.8 * len(self.file_list)) :]

    def __len__(self):
        return len(self.file_list)

    def pre_process_meshes(self):
        print("Pre-processing meshes...")
        for mesh_file in tqdm(self.file_list):
            mesh = meshio.read(mesh_file)
            vertices = mesh.points
            triangles = mesh.cells_dict["triangle"]
            mesh_data = MeshData(vertices, triangles)
            mesh_data.save(mesh_file[:-4] + ".npz")

    def load_pre_processed_mesh(self):
        print("Loading pre-processed meshes...")
        self.mesh_data_list = []
        for mesh_file in tqdm(self.file_list):
            mesh_data = MeshData()
            mesh_data.load(mesh_file[:-4] + ".npz")
            mesh_data.to_torch()
            self.mesh_data_list.append(mesh_data)

    def __getitem__(self, idx):
        mesh_data = self.mesh_data_list[idx]
        return mesh_data
