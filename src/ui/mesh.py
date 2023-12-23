import numpy as np
from OpenGL.GL import *
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


def load_obj(path, scale=1.0):
    mesh = meshio.read(path)
    vertices = mesh.points
    triangles = mesh.cells_dict["triangle"]
    vertices = vertices - np.mean(vertices, axis=0, keepdims=True)
    vertices = vertices / np.max(np.abs(vertices)) * scale
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
    def __init__(self, vertices, indices, program, camera):
        # Assuming each vertex entry in 'vertices' now includes position and texture coordinates and normal
        # e.g., [x, y, z, u, v, nx, ny, nz]
        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)
        self.gl_data = self.vertices[self.indices].reshape(-1)
        self.program = program
        self.camera = camera
        self.init_gl()
        self.displacement = [0.0, 0.0, 0.0]

    def init_gl(self):
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.PBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.PBO)
        glBufferData(GL_ARRAY_BUFFER, self.gl_data, GL_STATIC_DRAW)
        # # Position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), ctypes.c_void_p(0)
        )
        # Texture coordinate attribute
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(
            1,
            2,
            GL_FLOAT,
            GL_FALSE,
            8 * sizeof(GLfloat),
            ctypes.c_void_p(3 * sizeof(GLfloat)),
        )

        # Normal attribute
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(
            2,
            3,
            GL_FLOAT,
            GL_FALSE,
            8 * sizeof(GLfloat),
            ctypes.c_void_p(5 * sizeof(GLfloat)),
        )

        # create texture
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        # set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)

    def render(self):
        glUseProgram(self.program)
        texture_loc = glGetUniformLocation(self.program, "texture_diffuse")
        glUniform1i(texture_loc, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        displacement_loc = glGetUniformLocation(self.program, "displacement")
        glUniform3fv(displacement_loc, 1, self.displacement)
        self.camera.update_uniform(self.program)
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.PBO)
        glDrawArrays(GL_TRIANGLES, 0, 3 * len(self.indices))
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)

    def set_texture(self, img: np.ndarray):
        glBindTexture(GL_TEXTURE_2D, self.texture)
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
        glBindTexture(GL_TEXTURE_2D, 0)
        self.update_texture(img)

    def update_texture(self, img: np.ndarray):
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(
            GL_TEXTURE_2D, 0, 0, 0, img.shape[0], img.shape[1], GL_RGB, GL_FLOAT, img
        )
        glBindTexture(GL_TEXTURE_2D, 0)

    def __del__(self):
        glDeleteVertexArrays(1, [self.VAO])
        glDeleteBuffers(1, [self.PBO])
