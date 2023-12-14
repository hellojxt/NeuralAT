import os
import meshio
import numpy as np
from .bempp import BEMModel
from .mesh_process import tetra_from_mesh, update_triangle_normals
from scipy.spatial import KDTree
from .fem import FEMmodel, LOBPCG_solver, Material, MatSet


class SoundObj:
    def __init__(self, mesh_path):
        (
            self.vertices,
            self.tets,
            self.surf_vertices,
            self.surf_triangles,
        ) = tetra_from_mesh(mesh_path)
        self.bbox_min = self.vertices.min(axis=0)
        self.bbox_max = self.vertices.max(axis=0)
        self.surf_normals = update_triangle_normals(
            self.surf_vertices, self.surf_triangles
        )

    def normalize(self, scale=1.0):
        self.vertices = (self.vertices - (self.bbox_max + self.bbox_min) / 2) / (
            self.bbox_max - self.bbox_min
        ).max()
        self.vertices = self.vertices * scale
        self.surf_vertices = (
            self.surf_vertices - (self.bbox_max + self.bbox_min) / 2
        ) / (self.bbox_max - self.bbox_min).max()

        self.surf_vertices = self.surf_vertices * scale
        self.bbox_min = self.vertices.min(axis=0)
        self.bbox_max = self.vertices.max(axis=0)

    def modal_analysis(self, k=32, material=Material(MatSet.Plastic)):
        self.fem_model = FEMmodel(self.vertices, self.tets, material)
        eigenvalues, eigenvectors = LOBPCG_solver(
            self.fem_model.stiffness_matrix, self.fem_model.mass_matrix, k
        )
        eigenvectors = eigenvectors.reshape(-1, 3, k)
        kd_tree = KDTree(self.vertices)
        _, surf_points_index = kd_tree.query(self.surf_vertices)
        surf_eigenvecs = eigenvectors[surf_points_index]
        self.modes = surf_eigenvecs.reshape(-1, 3, k)
        self.eigenvalues = eigenvalues

    def get_triangle_neumann(self, mode_id):
        vertex_modes = self.modes[:, :, mode_id]
        triangle_neumann = vertex_modes[self.surf_triangles].mean(axis=1)
        triangle_neumann = (triangle_neumann * self.surf_normals).sum(axis=1)
        return triangle_neumann

    def get_triangle_neumanns(self, mode_ids):
        data = np.zeros((self.surf_triangles.shape[0], len(mode_ids)))
        for i, mode_id in enumerate(mode_ids):
            data[:, i] = self.get_triangle_neumann(mode_id)
        return data

    def get_frequencies(self):
        return self.eigenvalues**0.5 / (2 * np.pi)

    def get_frequency(self, mode_id):
        return self.eigenvalues[mode_id] ** 0.5 / (2 * np.pi)

    def get_omega(self, mode_id):
        return self.eigenvalues[mode_id] ** 0.5

    def get_wave_number(self, mode_id):
        return self.get_omega(mode_id) / 343.2

    def solve_by_BEM(self, mode_id, tol=1e-6, maxiter=2000):
        triangle_neumann = self.get_triangle_neumann(mode_id)
        bem_model = BEMModel(
            self.surf_vertices,
            self.surf_triangles,
            self.get_wave_number(mode_id),
        )
        residual = bem_model.boundary_equation_solve(triangle_neumann, tol, maxiter)
        if len(residual) == 0:
            return bem_model, 0
        return bem_model, residual[-1]


def solve_points_dirichlet(
    vertices, triangles, neumann, dirichlet, wave_number, points
):
    bem_model = BEMModel(vertices, triangles, wave_number)
    bem_model.set_dirichlet(dirichlet)
    bem_model.set_neumann(neumann)
    return bem_model.potential_solve(points)
