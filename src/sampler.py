from torch.utils.cpp_extension import load as load_cuda
import os
from glob import glob
import torch

class CUDA_MODULE:
    _module = None

    @staticmethod
    def get(name):
        if CUDA_MODULE._module is None:
            CUDA_MODULE.load()
        return getattr(CUDA_MODULE._module, name)

    @staticmethod
    def load(Debug=False, MemoryCheck=False, Verbose=False):
        src_dir = os.path.dirname(os.path.abspath(
            __file__)) + "/cuda"
        os.environ['TORCH_EXTENSIONS_DIR'] = os.path.join(src_dir, 'build')
        cflags = '--extended-lambda --expt-relaxed-constexpr '
        if Debug:
            cflags += '-G -g'
            cflags += ' -DDEBUG'
        else:
            cflags += '-O3'
            cflags += ' -DNDEBUG'
        if MemoryCheck:
            cflags += ' -DMEMORY_CHECK'
        cuda_files = glob(src_dir + '/*.cu')
        include_paths = [src_dir + '/include']
        CUDA_MODULE._module = load_cuda(name='CUDA_MODULE',
                                        sources=cuda_files,
                                        extra_include_paths=include_paths,
                                        extra_cuda_cflags=[cflags], 
                                        verbose=Verbose)
        return CUDA_MODULE._module

CUDA_MODULE.load(Debug=True, MemoryCheck=True, Verbose=True)


def uniform_sample(vertices, triangles, num_samples):
    """
    Uniformly sample points on the surface of a mesh.
    Args:
        vertices: (V, 3) float32 tensor of vertices
        triangles: (T, 3) int32 tensor of vertex indices
        num_samples: int, number of samples to generate
    Returns:
        points: (num_samples, 3) float32 tensor of sampled points
    """
    assert vertices.is_cuda
    points = torch.zeros((num_samples, 3), dtype=torch.float32, device=vertices.device)
    points_normals = torch.zeros((num_samples, 3), dtype=torch.float32, device=vertices.device)
    CUDA_MODULE.get('uniform_sample')(vertices, triangles, num_samples, points, points_normals)
    return points, points_normals
