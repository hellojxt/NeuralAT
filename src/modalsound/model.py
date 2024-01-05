import os
import meshio
import numpy as np
from .bempp import BEMModel
from .mesh_process import tetra_from_mesh, update_triangle_normals
from scipy.spatial import KDTree
from .fem import FEMmodel, LOBPCG_solver, Material, MatSet
from ..cuda_imp import multipole
import torch
from numba import njit


def SNR(ground_truth, prediction):
    ground_truth = np.abs(ground_truth)
    prediction = np.abs(prediction)
    return 10 * np.log10(
        (ground_truth**2).mean() / ((ground_truth - prediction) ** 2).mean()
    )


from skimage.metrics import structural_similarity as ssim


def complex_ssim(x, y):
    x = np.abs(x)
    y = np.abs(y)
    max_val = x.max()
    min_val = x.min()
    return ssim(x, y, data_range=max_val - min_val)


class MultipoleModel:
    def __init__(self, x0, n0, k, M):
        self.x0 = torch.tensor(x0).float().cuda()
        self.n0 = torch.tensor(n0).float().cuda()
        self.k = k
        self.M = M

    def solve_dirichlet(self, points):
        if isinstance(points, np.ndarray):
            points = torch.tensor(points).float().cuda().reshape(-1, 3)
        return multipole(self.x0, self.n0, points, points, self.k, self.M, False)

    def solve_neumann(self, points, normals):
        if isinstance(points, np.ndarray):
            points = torch.tensor(points).float().cuda().reshape(-1, 3)
        if isinstance(normals, np.ndarray):
            normals = torch.tensor(normals).float().cuda().reshape(-1, 3)
        return multipole(self.x0, self.n0, points, normals, self.k, self.M, True)


@njit()
def unit_sphere_surface_points(res):
    # r = 0.5
    points = np.zeros((2 * res, res, 3))
    phi_spacing = 2 * np.pi / (res * 2 - 1)
    theta_spacing = np.pi / (res - 1)
    for i in range(2 * res):
        for j in range(res):
            phi = phi_spacing * i
            theta = theta_spacing * j
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points[i, j] = [x, y, z]
    return points * 0.5


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


def get_mesh_center(vertices):
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    return (bbox_max + bbox_min) / 2


def get_mesh_size(vertices):
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    return (bbox_max - bbox_min).max()


def get_spherical_surface_points(vertices, scale=2):
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    points = unit_sphere_surface_points(32)
    points = points.reshape(-1, 3)
    points = points * get_mesh_size(vertices) * scale + get_mesh_center(vertices)
    points = torch.tensor(points).float().cuda()
    return points


class MeshObj:
    def __init__(self, mesh_path, scale=0.15):
        mesh = meshio.read(mesh_path)
        self.vertices = mesh.points
        bbox_min = self.vertices.min(axis=0)
        bbox_max = self.vertices.max(axis=0)
        self.vertices = (
            (self.vertices - (bbox_max + bbox_min) / 2)
            / (bbox_max - bbox_min).max()
            * scale
        )
        self.triangles = mesh.cells_dict["triangle"]
        self.triangles_normal = update_normals(self.vertices, self.triangles)
        self.triangles_center = self.vertices[self.triangles].mean(axis=1)
        self.bbox_min = self.vertices.min(axis=0)
        self.bbox_max = self.vertices.max(axis=0)
        self.center = (self.bbox_max + self.bbox_min) / 2
        self.size = (self.bbox_max - self.bbox_min).max()

    def spherical_surface_points(self, scale=2):
        points = unit_sphere_surface_points(32)
        points = points.reshape(-1, 3)
        points = points * self.size * scale + self.center
        points = torch.tensor(points).float().cuda()
        return points


class ModalSoundObj:
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

    def spherical_surface_points(self, scale=2):
        points = unit_sphere_surface_points(32)
        points = points.reshape(-1, 3)
        points = points * self.size * scale + self.center
        points = torch.tensor(points).float().cuda()
        return points

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
        self.size = (self.bbox_max - self.bbox_min).max()
        self.center = (self.bbox_max + self.bbox_min) / 2

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
