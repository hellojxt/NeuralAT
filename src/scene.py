import os
import numpy as np
import meshio


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


class SimpleSoundObject:
    def __init__(self, mesh_path):
        mesh = meshio.read(mesh_path)
        self.vertices = mesh.points
        self.triangles = mesh.cells_dict["triangle"]

    def translate(self, x, y, z):
        self.vertices += np.array([x, y, z])

    def scale(self, scale_factor):
        self.vertices *= scale_factor

    def set_neumann(self, neumann):
        self.triangles_neumann = np.ones(len(self.triangles)) * neumann


class SoundObjList:
    def __init__(self, mesh_list):
        self.vertices = None
        self.triangles = None
        self.triangles_neumann = None
        for simple_sound_obj in mesh_list:
            c_vertices = simple_sound_obj.vertices
            c_triangles = simple_sound_obj.triangles
            c_triangles_neumann = simple_sound_obj.triangles_neumann
            if self.vertices is None:
                self.vertices = c_vertices
                self.triangles = c_triangles
                self.triangles_neumann = c_triangles_neumann
            else:
                c_triangles = c_triangles + len(self.vertices)
                self.vertices = np.concatenate([self.vertices, c_vertices])
                self.triangles = np.concatenate([self.triangles, c_triangles])
                self.triangles_neumann = np.concatenate(
                    [self.triangles_neumann, c_triangles_neumann]
                )

        self.triangles_normal = update_normals(self.vertices, self.triangles)
        self.vertices_normal = update_vertex_normals(
            self.vertices, self.triangles, self.triangles_normal
        )
        self.bbox_min = self.vertices.min(axis=0)
        self.bbox_max = self.vertices.max(axis=0)
