import os
import numpy as np
from torch.utils.cpp_extension import load
import torch
from ..mesh_loader import tetra_from_mesh


def scipy2torch(M, device="cuda"):
    device = torch.device(device)
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long().to(device)
    values = torch.from_numpy(M.data).to(device)
    shape = torch.Size(M.shape)
    M_torch = torch.sparse_coo_tensor(indices, values, shape, device=device)
    return M_torch.coalesce()


def LOBPCG_solver(stiff_matrix, mass_matrix, k):
    vals, vecs = torch.lobpcg(stiff_matrix, k, mass_matrix, tracker=None, largest=False)
    return vals.cpu().numpy()[6:], vecs.cpu().numpy()[:, 6:]


class MatSet:
    Ceramic = 2700, 7.2e10, 0.19, 6, 1e-7
    Glass = 2600, 6.2e10, 0.20, 1, 1e-7
    Wood = 750, 1.1e10, 0.25, 60, 2e-6
    Plastic = 1070, 1.4e9, 0.35, 30, 1e-6
    Iron = 8000, 2.1e11, 0.28, 5, 1e-7
    Polycarbonate = 1190, 2.4e9, 0.37, 0.5, 4e-7
    Steel = 7850, 2.0e11, 0.29, 5, 3e-8
    Tin = 7265, 5e10, 0.325, 2, 3e-8


class Material(object):
    def __init__(self, material):
        self.density, self.youngs, self.poison, self.alpha, self.beta = material


cuda_dir = os.path.dirname(__file__) + "/cuda"
cuda_include_dir = cuda_dir + "/include"
os.environ["TORCH_EXTENSIONS_DIR"] = cuda_dir + "/build"
src_file = cuda_dir + "/computeMatrix.cu"
cuda_module = load(
    name="matrixAssemble",
    sources=[src_file],
    extra_include_paths=[cuda_include_dir],
    # extra_cuda_cflags=['-O3'],
    # extra_cuda_cflags=['-G -g'],
    #    verbose=True,
)


class FEMmodel:
    def __init__(self, vertices, tets, material=Material(MatSet.Plastic)):
        self.vertices = vertices.astype(np.float32)
        self.tets = tets.astype(np.int32)
        self.material = material
        self.stiffness_matrix_, self.mass_matrix_ = None, None

    @property
    def mass_matrix(self):
        if self.mass_matrix_ is None:
            values = torch.zeros(
                12 * 12 * self.tets.shape[0], dtype=torch.float32
            ).cuda()
            rows = torch.zeros(12 * 12 * self.tets.shape[0], dtype=torch.int32).cuda()
            cols = torch.zeros(12 * 12 * self.tets.shape[0], dtype=torch.int32).cuda()
            vertices_ = (
                torch.from_numpy(self.vertices)
                .to(torch.float32)
                .reshape(-1)
                .contiguous()
                .cuda()
            )
            tets_ = (
                torch.from_numpy(self.tets)
                .to(torch.int32)
                .reshape(-1)
                .contiguous()
                .cuda()
            )
            cuda_module.assemble_mass_matrix(
                vertices_, tets_, values, rows, cols, self.material.density
            )
            indices = torch.stack([rows, cols], dim=0).long()
            shape = torch.Size([3 * self.vertices.shape[0], 3 * self.vertices.shape[0]])
            self.mass_matrix_ = torch.sparse_coo_tensor(indices, values, shape)
        return self.mass_matrix_

    @property
    def stiffness_matrix(self):
        if self.stiffness_matrix_ is None:
            values = torch.zeros(
                12 * 12 * self.tets.shape[0], dtype=torch.float32
            ).cuda()
            rows = torch.zeros(12 * 12 * self.tets.shape[0], dtype=torch.int32).cuda()
            cols = torch.zeros(12 * 12 * self.tets.shape[0], dtype=torch.int32).cuda()
            vertices_ = (
                torch.from_numpy(self.vertices)
                .to(torch.float32)
                .reshape(-1)
                .contiguous()
                .cuda()
            )
            tets_ = (
                torch.from_numpy(self.tets)
                .to(torch.int32)
                .reshape(-1)
                .contiguous()
                .cuda()
            )
            cuda_module.assemble_stiffness_matrix(
                vertices_,
                tets_,
                values,
                rows,
                cols,
                self.material.youngs,
                self.material.poison,
            )
            indices = torch.stack([rows, cols], dim=0).long()
            shape = torch.Size([3 * self.vertices.shape[0], 3 * self.vertices.shape[0]])
            self.stiffness_matrix_ = torch.sparse_coo_tensor(indices, values, shape)
        return self.stiffness_matrix_


import pymesh


class SoundObj:
    def __init__(self, mesh_file):
        self.origin_mesh = pymesh.load_mesh(mesh_file)
        self.mesh_file = mesh_file
        # print("original mesh data")
        # print("vertices: ", self.origin_mesh.vertices.shape)
        # print("faces: ", self.origin_mesh.faces.shape)
        bbox_min, bbox_max = self.origin_mesh.bbox
        # print("bbox_min: ", bbox_min)
        # print("bbox_max: ", bbox_max)
        self.cell_size = (bbox_max - bbox_min).max() / 30
        # print("target cell_size: ", self.cell_size)

    def tetrahedralize(self, engine="cgal"):
        self.tet_vertices, self.tet_voxels = tetra_from_mesh(self.mesh_file)

    def modal_analysis(self, k=200, material=Material(MatSet.Plastic)):
        self.fem_model = FEMmodel(self.tet_vertices, self.tet_voxels, material)
        self.eigenvalues, self.eigenvectors = LOBPCG_solver(
            self.fem_model.stiffness_matrix, self.fem_model.mass_matrix, k + 6
        )

    def show_frequencys(self, num=20):
        print((self.eigenvalues**0.5 / (2 * np.pi))[:num])
