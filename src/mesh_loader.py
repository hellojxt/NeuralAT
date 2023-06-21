import torch
import meshio
import subprocess
import os
from glob import glob


def tetra_surf_from_triangle_mesh(input_mesh, output_dir, log=False):
    """
    Create a tetrahedral mesh from a triangle mesh. Then read the surface mesh.
    input: the path to the input mesh
    output: the directory to store the output mesh
    """
    mesh = meshio.read(input_mesh)
    # check if the mesh is a triangle mesh
    assert mesh.cells[0].type == "triangle"
    # convert the triangle mesh to a tetrahedral mesh
    # need to install FloatTetWild first
    result = subprocess.run(
        ["FloatTetwild_bin", "-i", input_mesh, "--max-threads", "8", "--coarsen"],
        capture_output=True,
        text=True,
    )
    if log:
        print(result.stdout, result.stderr)
    # copy the output mesh to the output directory
    os.makedirs(output_dir, exist_ok=True)
    output_mesh = os.path.join(output_dir, os.path.basename(input_mesh))
    subprocess.run(
        ["cp", input_mesh + "__sf.obj", output_mesh.replace(".obj", ".sf.obj")]
    )
    # remove input_mesh_*
    for f in glob(input_mesh + "_*"):
        os.remove(f)


def load_surface_mesh(input_mesh):
    surf_mesh = meshio.read(input_mesh)
    # remove the intermediate file
    vertices = torch.Tensor(surf_mesh.points).cuda()
    triangles = torch.Tensor(surf_mesh.cells[0].data).long().cuda()
    return vertices, triangles
