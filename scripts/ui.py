import sys

sys.path.append("./")
import imgui
from src.ui.ui import UI
from src.ui.mesh import Mesh, load_obj
import numpy as np

width, height = 800, 800

ui = UI(width, height, shader="mesh")

bunny = Mesh(*load_obj("dataset/bunny.obj"), ui.program, ui.camera)
texture = np.ones((512, 512, 3), dtype=np.float32)
bunny.set_texture(texture)

sphere = Mesh(*load_obj("dataset/sphere.obj", scale=0.2), ui.program, ui.camera)
sphere.set_texture(texture)
sphere.displacement = [0.0, 1.5, 0.0]

while not ui.should_close():
    ui.begin_imgui()
    imgui.begin("3D Vector Control")

    # Create sliders for each component of the vector
    changed, sphere.displacement[0] = imgui.slider_float(
        "X", sphere.displacement[0], min_value=-2.0, max_value=2.0
    )
    changed, sphere.displacement[1] = imgui.slider_float(
        "Y", sphere.displacement[1], min_value=-2.0, max_value=2.0
    )
    changed, sphere.displacement[2] = imgui.slider_float(
        "Z", sphere.displacement[2], min_value=-2.0, max_value=2.0
    )

    # Display the current vector values
    imgui.text(
        f"Vector: ({sphere.displacement[0]:.2f}, {sphere.displacement[1]:.2f}, {sphere.displacement[2]:.2f})"
    )

    imgui.end()
    ui.end_imgui()

    ui.begin_gl()
    bunny.update_texture(texture * (np.sin(ui.current * 4) * 0.4 + 0.6))
    bunny.render()
    sphere.render()
    ui.end_gl()
    # sphere.render()
ui.close()
