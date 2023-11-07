import os
import sys
from .kleinpat import read_mode_data
import meshio
import numpy as np
from .bempp import BEMModel


def update_normals(vertices, triangles):
    """
    vertices: (n, 3)
    triangles: (m, 3)
    """
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return normals


def update_vertex_normals(vertices, triangles, triangle_normals):
    """
    vertices: (n, 3)
    triangles: (m, 3)
    triangle_normals: (m, 3)
    """
    vertex_normals = np.zeros_like(vertices)
    for i in range(len(triangles)):
        vertex_normals[triangles[i, 0]] += triangle_normals[i]
        vertex_normals[triangles[i, 1]] += triangle_normals[i]
        vertex_normals[triangles[i, 2]] += triangle_normals[i]
    vertex_normals = vertex_normals / np.linalg.norm(
        vertex_normals, axis=1, keepdims=True
    )
    return vertex_normals


class ModalSoundObject:
    def __init__(self, data_dir):
        basename = os.path.basename(data_dir)
        self.mode_file = os.path.join(data_dir, basename + "_surf.modes")
        self.model_file = os.path.join(data_dir, basename + ".tet.obj")
        self.material_file = os.path.join(data_dir, basename + "_material.txt")
        self.ffat_dir = os.path.join(data_dir, basename + "_ffat_maps")
        self.dirichlet_dir = os.path.join(data_dir, basename + "_dirichlet")
        if not os.path.exists(self.dirichlet_dir):
            os.makedirs(self.dirichlet_dir)

        self.nDOF, self.nModes, self.omega_squared, self.modes = read_mode_data(
            self.mode_file
        )

        mesh = meshio.read(self.model_file)
        self.vertices = mesh.points
        self.triangles = mesh.cells_dict["triangle"]
        self.triangles_normal = update_normals(self.vertices, self.triangles)
        self.vertices_normal = update_vertex_normals(
            self.vertices, self.triangles, self.triangles_normal
        )
        self.bbox_min = self.vertices.min(axis=0)
        self.bbox_max = self.vertices.max(axis=0)

        material = np.loadtxt(self.material_file)
        self.rho, self.youngs, self.poisson, self.alpha, self.beta = material

    def scaled_bbox(self, scale_factor):
        bbox_center = (self.bbox_min + self.bbox_max) / 2
        bbox_size = self.bbox_max - self.bbox_min
        bbox_size = bbox_size * scale_factor
        bbox_min = bbox_center - bbox_size / 2
        bbox_max = bbox_center + bbox_size / 2
        return bbox_min, bbox_max

    def get_triangle_neumann(self, mode_id):
        vertex_modes = self.modes[mode_id].reshape(-1, 3)
        triangle_neumann = vertex_modes[self.triangles].mean(axis=1)
        # triangle_neumann = (triangle_neumann**2).sum(axis=1)
        # triangle_neumann = triangle_neumann / triangle_neumann.max()
        triangle_neumann = (triangle_neumann * self.triangles_normal).sum(axis=1)
        return triangle_neumann

    def get_vertex_dirichlet(self, mode_id, force_recompute=False):
        dirichlet_real_path = os.path.join(self.dirichlet_dir, f"{mode_id}_real.npy")
        dirichlet_imag_path = os.path.join(self.dirichlet_dir, f"{mode_id}_imag.npy")
        dirichlet_error_path = os.path.join(self.dirichlet_dir, f"{mode_id}_error.txt")
        dirichlet_cost_time_path = os.path.join(
            self.dirichlet_dir, f"{mode_id}_cost_time.txt"
        )
        import time

        if (
            not os.path.exists(dirichlet_real_path)
            or not os.path.exists(dirichlet_imag_path)
            or not os.path.exists(dirichlet_error_path)
            or not os.path.exists(dirichlet_cost_time_path)
            or force_recompute
        ):
            triangle_neumann = self.get_triangle_neumann(mode_id)
            bem_model = BEMModel(
                self.vertices,
                self.triangles,
                self.get_wave_number(mode_id),
            )
            start = time.time()
            residual = bem_model.boundary_equation_solve(triangle_neumann)
            end = time.time()
            dirichlet = bem_model.get_dirichlet_coeff().reshape(-1, 1)
            np.save(dirichlet_real_path, dirichlet.real)
            np.save(dirichlet_imag_path, dirichlet.imag)
            np.savetxt(dirichlet_error_path, residual)
            np.savetxt(dirichlet_cost_time_path, [end - start])
        dirichlet_real_gt = np.load(dirichlet_real_path)
        dirichlet_imag_gt = np.load(dirichlet_imag_path)
        dirichlet_error = np.loadtxt(dirichlet_error_path)
        cost_time = np.loadtxt(dirichlet_cost_time_path)
        return dirichlet_real_gt, dirichlet_imag_gt, dirichlet_error, cost_time

    def get_triangle_center(self):
        triangle_center = self.vertices[self.triangles].mean(axis=1)
        return triangle_center

    def get_omega(self, mode_id):
        return (self.omega_squared[mode_id] / self.rho) ** 0.5

    def get_wave_number(self, mode_id):
        return self.get_omega(mode_id) / 343.2
