import numpy as np
import glm
from OpenGL.GL import *
import glfw
import imgui


class Camera:
    def __init__(self, position, target_position, light_direction, light_color):
        self.position = glm.vec3(position)
        self.target_position = glm.vec3(target_position)
        self.up = glm.vec3(0, 1, 0)
        self.distance = glm.length(self.position - self.target_position)
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
        self.light_direction = glm.vec3(light_direction)
        self.light_color = glm.vec3(light_color)

    def get_view_matrix(self):
        direction = glm.vec3(
            np.cos(self.vertical_angle) * np.sin(self.horizontal_angle),
            np.sin(self.vertical_angle),
            np.cos(self.vertical_angle) * np.cos(self.horizontal_angle),
        )
        camera_position = self.target_position - direction * self.distance
        return glm.lookAt(camera_position, self.target_position, self.up)

    def get_projection_matrix(self, width, height):
        return glm.perspective(
            glm.radians(self.field_of_view),
            width / height,
            self.near_plane,
            self.far_plane,
        )

    def orbit(self, xoffset, yoffset):
        self.horizontal_angle += self.orbit_speed * xoffset
        self.vertical_angle += self.orbit_speed * yoffset
        self.vertical_angle = max(
            -np.pi / 2, min(np.pi / 2, self.vertical_angle)
        )  # Clamp the vertical angle

    def zoom(self, yoffset):
        self.distance = max(1.0, self.distance - yoffset * self.zoom_speed)

    def pan(self, xoffset, yoffset):
        right = glm.vec3(
            np.sin(self.horizontal_angle - 3.14 / 2.0),
            0,
            np.cos(self.horizontal_angle - 3.14 / 2.0),
        )
        up = glm.cross(
            right,
            glm.vec3(
                np.cos(self.vertical_angle) * np.sin(self.horizontal_angle),
                np.sin(self.vertical_angle),
                np.cos(self.vertical_angle) * np.cos(self.horizontal_angle),
            ),
        )
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

    def update_uniform(self, program, width, height):
        view_loc = glGetUniformLocation(program, "view")
        projection_loc = glGetUniformLocation(program, "projection")
        model_loc = glGetUniformLocation(program, "model")
        lightDir_loc = glGetUniformLocation(program, "lightDir")
        lightColor_loc = glGetUniformLocation(program, "lightColor")
        view_pos = glGetUniformLocation(program, "viewPos")

        glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(self.get_view_matrix()))
        glUniformMatrix4fv(
            projection_loc,
            1,
            GL_FALSE,
            glm.value_ptr(self.get_projection_matrix(width, height)),
        )
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(glm.mat4(1.0)))
        glUniform3fv(lightDir_loc, 1, glm.value_ptr(self.light_direction))
        glUniform3fv(lightColor_loc, 1, glm.value_ptr(self.light_color))
        glUniform3fv(view_pos, 1, glm.value_ptr(self.position))
