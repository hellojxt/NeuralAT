import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
from OpenGL.GL.ARB.pixel_buffer_object import *
import imgui
from imgui.integrations.glfw import GlfwRenderer
import ctypes
from .camera import Camera
from .mesh import Mesh
from .gl_func import create_window, create_program
import numpy as np
import torch
import time
import os
import warnings
import glm


class UI:
    def __init__(self, width, height, name="Render"):
        self.width = width
        self.height = height
        self.window = create_window(width, height, name)
        self.impl = GlfwRenderer(self.window)
        # create shader, vao, texture
        self.program = create_program()
        self.current = time.time()
        self.duration = 0
        light_direction = glm.vec3(-0.2, -1.0, -0.3)
        light_color = glm.vec3(1.0, 1.0, 1.0)
        self.camera = Camera([0, 0, 5], [0, 0, 0], light_direction, light_color)
        glfw.set_cursor_pos_callback(self.window, self.camera.mouse_callback)
        glfw.set_scroll_callback(self.window, self.camera.scroll_callback)

        self.displacement = [0.0, 0.0, 0.0]

    def close(self):
        glDeleteProgram(self.program)
        self.impl.shutdown()
        glfw.destroy_window(self.window)
        glfw.terminate()

    def process_input(self):
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)

    def should_close(self):
        return glfw.window_should_close(self.window)

    def begin_frame(self):
        t = time.time()
        fps = 1.0 / (t - self.current)
        self.duration += t - self.current
        self.current = t
        imgui.new_frame()
        imgui.begin("Options")

        imgui.text("Time: {:.1f}".format(self.duration))
        imgui.text("FPS: {:.1f}".format(fps))

        changed, self.displacement[0] = imgui.slider_float(
            "X Displacement", self.displacement[0], -1.0, 1.0
        )
        if changed:
            displacement_loc = glGetUniformLocation(self.program, "displacement")
            glUniform3fv(displacement_loc, 1, self.displacement)

        changed, self.displacement[1] = imgui.slider_float(
            "Y Displacement", self.displacement[1], -1.0, 1.0
        )
        if changed:
            displacement_loc = glGetUniformLocation(self.program, "displacement")
            glUniform3fv(displacement_loc, 1, self.displacement)

        changed, self.displacement[2] = imgui.slider_float(
            "Z Displacement", self.displacement[2], -1.0, 1.0
        )
        if changed:
            displacement_loc = glGetUniformLocation(self.program, "displacement")
            glUniform3fv(displacement_loc, 1, self.displacement)

        imgui.end()
        imgui.render()
        imgui.end_frame()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.program)
        self.camera.update_uniform(self.program, self.width, self.height)

    def end_frame(self):
        self.impl.render(imgui.get_draw_data())
        self.impl.process_inputs()
        self.process_input()
        glfw.swap_buffers(self.window)
        glfw.poll_events()
