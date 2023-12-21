import sys

sys.path.append("./")

from src.ui.ui import UI
from src.ui.mesh import Mesh, load_obj
import numpy as np

width, height = 1280, 720

ui = UI(width, height)

bunny = Mesh(*load_obj("dataset/monopole/bunny/mesh.obj"), ui.program)
texture = np.ones((512, 512, 3), dtype=np.float32)
bunny.set_cpu_texture(texture)

sphere = Mesh(*load_obj("dataset/sphere.obj"), ui.program)
sphere.set_cpu_texture(texture)

while not ui.should_close():
    ui.begin_frame()
    bunny.update_cpu_texture(texture * (np.sin(ui.current * 4) * 0.4 + 0.6))
    bunny.render()
    sphere.render()
    ui.end_frame()

ui.close()
