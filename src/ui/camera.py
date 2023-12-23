import numpy as np
from OpenGL.GL import *
import glfw
import imgui


class Camera:
    def __init__(
        self,
        position,
        target_position,
        light_direction,
        light_color,
        window,
    ):
        self.position = np.array(position)
        self.target_position = np.array(target_position)
        self.up = np.array([0, 1, 0])
        self.distance = np.linalg.norm(self.position - self.target_position)
        self.horizontal_angle = 3.14
        self.vertical_angle = 0.0
        self.field_of_view = 45.0
        self.near_plane = 0.1
        self.far_plane = 1000.0
        self.zoom_speed = 0.1
        self.orbit_speed = 0.005
        self.pan_speed = 0.005
        self.last_x, self.last_y = 0, 0
        self.first_mouse = True
        self.light_direction = np.array(light_direction)
        self.light_color = np.array(light_color)
        self.window = window

    def get_view_matrix(self):
        direction = np.array(
            [
                np.cos(self.vertical_angle) * np.sin(self.horizontal_angle),
                np.sin(self.vertical_angle),
                np.cos(self.vertical_angle) * np.cos(self.horizontal_angle),
            ]
        )
        camera_position = self.target_position - direction * self.distance
        return self.look_at(camera_position, self.target_position, self.up)

    def get_projection_matrix(self, width, height):
        return self.perspective(
            np.radians(self.field_of_view),
            width / height,
            self.near_plane,
            self.far_plane,
        )

    def look_at(self, eye, center, up):
        f = self.normalize(center - eye)
        s = self.normalize(np.cross(up, f))
        u = np.cross(f, s)

        M = np.identity(4)
        M[0, :3] = s
        M[1, :3] = u
        M[2, :3] = -f

        T = np.identity(4)
        T[:3, 3] = -eye

        return np.dot(M, T).T

    def perspective(self, fov, aspect, near, far):
        f = 1.0 / np.tan(fov / 2)
        M = np.zeros((4, 4))
        M[0, 0] = f / aspect
        M[1, 1] = f
        M[2, 2] = (far + near) / (near - far)
        M[3, 2] = -1.0
        M[2, 3] = (2 * far * near) / (near - far)
        return M.T

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def orbit(self, xoffset, yoffset):
        self.horizontal_angle += self.orbit_speed * xoffset
        self.vertical_angle += self.orbit_speed * yoffset
        self.vertical_angle = max(
            -np.pi / 2, min(np.pi / 2, self.vertical_angle)
        )  # Clamp the vertical angle

    def zoom(self, yoffset):
        self.distance = max(1.0, self.distance - yoffset * self.zoom_speed)

    def pan(self, xoffset, yoffset):
        right = np.array(
            [
                np.sin(self.horizontal_angle - 3.14 / 2.0),
                0,
                np.cos(self.horizontal_angle - 3.14 / 2.0),
            ]
        )
        forward = np.array(
            [
                np.cos(self.vertical_angle) * np.sin(self.horizontal_angle),
                np.sin(self.vertical_angle),
                np.cos(self.vertical_angle) * np.cos(self.horizontal_angle),
            ]
        )
        up = np.cross(right, forward)
        self.target_position += right * xoffset * self.pan_speed
        self.target_position += up * yoffset * self.pan_speed

    def mouse_callback(self, window, xpos, ypos):
        if (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
            and not imgui.is_any_item_active()
        ):
            if self.first_mouse:
                self.last_x, self.last_y = xpos, ypos
                self.first_mouse = False

            xoffset = xpos - self.last_x
            yoffset = (
                self.last_y - ypos
            )  # reversed since y-coordinates go from bottom to top
            self.last_x, self.last_y = xpos, ypos

            self.orbit(xoffset, yoffset)
        else:
            self.first_mouse = True

    def scroll_callback(self, window, xoffset, yoffset):
        self.zoom(yoffset)

    def update_uniform(self, program):
        view_loc = glGetUniformLocation(program, "view")
        projection_loc = glGetUniformLocation(program, "projection")
        model_loc = glGetUniformLocation(program, "model")
        lightDir_loc = glGetUniformLocation(program, "lightDir")
        lightColor_loc = glGetUniformLocation(program, "lightColor")
        view_pos = glGetUniformLocation(program, "viewPos")

        glUniformMatrix4fv(
            view_loc, 1, GL_FALSE, self.get_view_matrix().astype(np.float32)
        )
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)
        glUniformMatrix4fv(
            projection_loc,
            1,
            GL_FALSE,
            self.get_projection_matrix(framebuffer_width, framebuffer_height).astype(
                np.float32
            ),
        )
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, np.identity(4, dtype=np.float32))
        glUniform3fv(lightDir_loc, 1, self.light_direction.astype(np.float32))
        glUniform3fv(lightColor_loc, 1, self.light_color.astype(np.float32))
        glUniform3fv(view_pos, 1, self.position.astype(np.float32))
