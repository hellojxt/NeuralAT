import numpy as np
from OpenGL.GL import *
import meshio
import torch


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


def load_obj(path, normalize=True):
    mesh = meshio.read(path)
    vertices = mesh.points
    triangles = mesh.cells_dict["triangle"]
    if normalize:
        vertices = vertices - np.mean(vertices, axis=0, keepdims=True)
        vertices = vertices / np.max(np.abs(vertices))
    normals = update_normals(vertices, triangles)
    vertex_normals = np.zeros_like(vertices)
    for i in range(triangles.shape[0]):
        vertex_normals[triangles[i]] += normals[i]
    vertex_normals = vertex_normals / np.linalg.norm(
        vertex_normals, axis=1, keepdims=True
    )
    # get texture coordinates
    if "texture" in mesh.point_data:
        tex_coords = mesh.point_data["texture"]
    else:
        tex_coords = np.zeros((vertices.shape[0], 2), dtype=np.float32)
    vertices = np.concatenate([vertices, tex_coords, vertex_normals], axis=1)
    return vertices, triangles


class Mesh:
    def __init__(self, vertices, indices, program):
        # Assuming each vertex entry in 'vertices' now includes position and texture coordinates and normal
        # e.g., [x, y, z, u, v, nx, ny, nz]
        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)
        self.VAO = None
        self.VBO = None
        self.EBO = None
        self.texture = None  # Handle for the texture
        self.setup_mesh()
        self.program = program

    def setup_mesh(self):
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(
            GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW
        )

        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW
        )

        # Position attribute
        glVertexAttribPointer(
            0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), ctypes.c_void_p(0)
        )
        glEnableVertexAttribArray(0)

        # Texture coordinate attribute
        glVertexAttribPointer(
            1,
            2,
            GL_FLOAT,
            GL_FALSE,
            8 * sizeof(GLfloat),
            ctypes.c_void_p(3 * sizeof(GLfloat)),
        )
        glEnableVertexAttribArray(1)

        # Normal attribute
        glVertexAttribPointer(
            2,
            3,
            GL_FLOAT,
            GL_FALSE,
            8 * sizeof(GLfloat),
            ctypes.c_void_p(5 * sizeof(GLfloat)),
        )
        glEnableVertexAttribArray(2)

        # create texture
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        # set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    def render(self):
        # Set texture uniform location (ideally done once after shader program is linked)
        texture_loc = glGetUniformLocation(self.program, "texture_diffuse")
        glUniform1i(texture_loc, 0)

        # Activate texture and bind it
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        # Bind VAO and draw
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, len(self.indices) * 3, GL_UNSIGNED_INT, None)

    def set_cpu_texture(self, img: np.ndarray):
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB32F,
            img.shape[0],
            img.shape[1],
            0,
            GL_RGB,
            GL_FLOAT,
            None,
        )
        self.update_cpu_texture(img)

    def update_cpu_texture(self, img: np.ndarray):
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(
            GL_TEXTURE_2D, 0, 0, 0, img.shape[0], img.shape[1], GL_RGB, GL_FLOAT, img
        )

    def __del__(self):
        glDeleteVertexArrays(1, [self.VAO])
        glDeleteBuffers(1, [self.VBO])
        glDeleteBuffers(1, [self.EBO])
