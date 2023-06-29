import numpy as np
import torch
from torch.utils.cpp_extension import load as load_cuda
import os
from glob import glob


class CUDA_MODULE:
    _module = None

    @staticmethod
    def get(name):
        if CUDA_MODULE._module is None:
            CUDA_MODULE.load()
        return getattr(CUDA_MODULE._module, name)

    @staticmethod
    def load(Debug=False, MemoryCheck=False, Verbose=False):
        src_dir = os.path.dirname(os.path.abspath(__file__)) + "/cuda"
        os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(src_dir, "build")
        cflags = ""
        if Debug:
            cflags += " -G -g"
            cflags += " -DDEBUG"
        else:
            cflags += " -O3"
            cflags += " -DNDEBUG"
        if MemoryCheck:
            cflags += " -DMEMORY_CHECK"
        cuda_files = [os.path.join(src_dir, "bind.cu")]
        include_paths = [src_dir, os.path.join(src_dir, "include")]
        CUDA_MODULE._module = load_cuda(
            name="BEM_CUDA",
            sources=cuda_files,
            extra_include_paths=include_paths,
            extra_cuda_cflags=[cflags],
            verbose=Verbose,
        )
        return CUDA_MODULE._module


CUDA_MODULE.load(Debug=False, MemoryCheck=False, Verbose=False)
boundary_operator_assembler = CUDA_MODULE.get("assemble_matrix")


def assemble_single_boundary_matrix(vertices, triangles, wave_number):
    matrix_cuda = torch.zeros(triangles.shape[0], triangles.shape[0]).cuda()
    boundary_operator_assembler(vertices, triangles, matrix_cuda, wave_number, False)
    return matrix_cuda


def assemble_double_boundary_matrix(vertices, triangles, wave_number):
    matrix_cuda = torch.zeros(triangles.shape[0], triangles.shape[0]).cuda()
    boundary_operator_assembler(vertices, triangles, matrix_cuda, wave_number, True)
    return matrix_cuda