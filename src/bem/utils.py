from numba import njit
import numpy as np


@njit()
def unit_cube_surface_points_(res):
    face_datas = np.array(
        [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    ).reshape(-1, 3, 3)
    points = np.zeros((6, res, res, 3))
    for face_idx, data in enumerate(face_datas):
        normal, w, h = data
        dw, dh = w / (res - 1), h / (res - 1)
        p0 = 0.5 * normal - 0.5 * w - 0.5 * h
        for i in range(res):
            for j in range(res):
                points[face_idx, i, j] = p0 + i * dw + j * dh
    return points


def ffat_cube_points(bbox_min, bbox_max, res):
    points = unit_cube_surface_points_(res)
    points = points.reshape(-1, 3)
    points = points * (bbox_max - bbox_min) + (bbox_max + bbox_min) / 2
    points = points.reshape(6 * res, res, 3)
    return points
