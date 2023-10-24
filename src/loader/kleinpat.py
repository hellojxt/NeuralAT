import numpy as np
import struct
from . import ffat_map_pb2


def read_mode_data(filename):
    with open(filename, "rb") as fin:
        # Read the size of the problem and the number of modes
        nDOF, nModes = struct.unpack("ii", fin.read(8))

        # Read the eigenvalues
        omega_squared = np.fromfile(fin, dtype=np.float64, count=nModes)

        # Read the eigenvectors
        modes = []
        for i in range(nModes):
            mode_data = np.fromfile(fin, dtype=np.float64, count=nDOF)
            modes.append(mode_data)

    return nDOF, nModes, np.asarray(omega_squared), np.asarray(modes)


def load_ffat_map(filename):
    ffat_map = ffat_map_pb2.ffat_map_double()

    with open(filename, "rb") as f:
        ffat_map.ParseFromString(f.read())

    map_3_out = ffat_map.map
    map_1 = map_3_out.shells

    # Parse cellsize
    cell_size = map_1.cellsize

    # Parse lowcorners
    low_corners = [
        [map_1.lowcorners.item[i].item[j] for j in range(3)]
        for i in range(len(map_1.lowcorners.item))
    ]

    # Parse n_elements
    n_elements = [
        [map_1.n_elements.item[0], map_1.n_elements.item[1]]
        for i in range(len(map_1.n_elements.item))
    ]

    # Parse strides
    strides = list(map_1.strides.item)

    # Parse center
    center = list(map_1.center.item)

    # Parse bboxlow
    bbox_low = list(map_1.bboxlow.item)

    # Parse bboxtop
    bbox_top = list(map_1.bboxtop.item)

    # Parse k
    k = map_3_out.k

    # Parse center
    center_3 = list(map_3_out.center.item)

    # Parse is_compressed
    is_compressed = map_3_out.is_compressed

    # Parse psi
    psi = [
        [item for item in map_3_out.psi.item[i].item]
        for i in range(len(map_3_out.psi.item))
    ]

    mode_id = map_3_out.modeid

    return (
        cell_size,
        low_corners,
        n_elements,
        strides,
        center,
        bbox_low,
        bbox_top,
        k,
        center_3,
        is_compressed,
        psi,
        mode_id,
    )
