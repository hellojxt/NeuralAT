import sys

sys.path.append("./")
from src.loader.model import ModalSoundObject

for obj_id in [0, 1, 2, 3]:
    sound_object = ModalSoundObject(f"dataset/0000{obj_id}")
    for mode_id in range(5):
        sound_object.get_vertex_dirichlet(mode_id, force_recompute=True)
