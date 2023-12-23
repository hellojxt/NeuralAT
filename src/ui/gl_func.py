import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
from OpenGL.GL.ARB.pixel_buffer_object import *
import imgui
import os


def framebuffer_size_callback(window, width, height):
    glViewport(0, 0, width, height)


def create_window(width, height, name="Render"):
    # initialize glfw
    if not glfw.init():
        print("Failed to initialize GLFW")
        exit()

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(width, height, name, None, None)

    if not window:
        print("Failed to create window")
        glfw.terminate()
        exit()

    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glfw.swap_interval(0)

    # initialize imgui
    imgui.create_context()
    return window


def create_program(flag="mesh"):
    file_path = os.path.dirname(os.path.realpath(__file__))
    vertex_path = file_path + f"/shader/{flag}.vert"
    fragment_path = file_path + f"/shader/{flag}.frag"
    with open(vertex_path, "r") as f:
        vertex_shader = f.read()

    with open(fragment_path, "r") as f:
        fragment_shader = f.read()

    vertex = OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
    fragment = OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    return OpenGL.GL.shaders.compileProgram(vertex, fragment, validate=False)
