import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
from OpenGL.GL.ARB.pixel_buffer_object import *
import imgui
from imgui.integrations.glfw import GlfwRenderer
import ctypes
from .camera import Camera
from .mesh import Mesh

import numpy as np
import torch
import time
import os
import warnings


def framebuffer_size_callback(window, width, height):
    glViewport(0, 0, width, height)


def create_window(width, height, name="Render"):
    # initialize glfw
    if not glfw.init():
        print("Failed to initialize GLFW")
        exit()

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(width, height, name, None, None)

    if not window:
        print("Failed to create window")
        glfw.terminate()
        exit()

    glfw.make_context_current(window)

    glViewport(0, 0, width, height)
    glEnable(GL_DEPTH_TEST)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glfw.swap_interval(0)

    # initialize imgui
    imgui.create_context()
    return window


def create_program():
    file_path = os.path.dirname(os.path.realpath(__file__))
    vertex_path = file_path + "/shader/mesh.vert"
    fragment_path = file_path + "/shader/mesh.frag"
    with open(vertex_path, "r") as f:
        vertex_shader = f.read()

    with open(fragment_path, "r") as f:
        fragment_shader = f.read()

    vertex = OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
    fragment = OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    return OpenGL.GL.shaders.compileProgram(vertex, fragment)
