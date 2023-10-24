import os
import sys
from .kleinpat import read_mode_data
import meshio
import numpy as np


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


class ModalSoundObject:
    def __init__(self, data_dir):
        basename = os.path.basename(data_dir)
        self.mode_file = os.path.join(data_dir, basename + "_surf.modes")
        self.model_file = os.path.join(data_dir, basename + ".tet.obj")
        self.material_file = os.path.join(data_dir, basename + "_material.txt")
        self.ffat_dir = os.path.join(data_dir, basename + "_ffat_maps")

        self.nDOF, self.nModes, self.omega_squared, self.modes = read_mode_data(
            self.mode_file
        )

        mesh = meshio.read(self.model_file)
        self.vertices = mesh.points
        self.triangles = mesh.cells_dict["triangle"]
        self.triangles_normal = update_normals(self.vertices, self.triangles)
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

    def get_triangle_center(self):
        triangle_center = self.vertices[self.triangles].mean(axis=1)
        return triangle_center

    def get_omega(self, mode_id):
        return (self.omega_squared[mode_id] / self.rho) ** 0.5

    def get_wave_number(self, mode_id):
        return self.get_omega(mode_id) / 343.2
