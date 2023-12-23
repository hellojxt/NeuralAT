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
import time


class UI:
    def __init__(self, width, height, name="Render", shader="mesh"):
        self.width = width
        self.height = height
        self.window = create_window(width, height, name)
        self.impl = GlfwRenderer(self.window)
        # create shader, vao, texture
        self.program = create_program(shader)
        self.current = time.time()
        self.duration = 0
        light_direction = [-0.2, -1.0, -0.3]
        light_color = [1.0, 1.0, 1.0]
        self.camera = Camera(
            [5, 0, 5],
            [0, 0, 0],
            light_direction,
            light_color,
            self.window,
        )
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

    def begin_imgui(self):
        t = time.time()
        fps = 1.0 / (t - self.current)
        self.duration += t - self.current
        self.current = t
        imgui.new_frame()
        imgui.begin("Options")

        imgui.text("Time: {:.1f}".format(self.duration))
        imgui.text("FPS: {:.1f}".format(fps))

    def end_imgui(self):
        imgui.end()
        imgui.render()
        imgui.end_frame()

    def begin_gl(self):
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)
        glViewport(0, 0, framebuffer_width, framebuffer_height)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def end_gl(self):
        self.impl.render(imgui.get_draw_data())
        self.impl.process_inputs()
        self.process_input()
        glfw.swap_buffers(self.window)
        glfw.poll_events()
