from torch.utils.cpp_extension import load as load_cuda
import os
from glob import glob
import torch
from .BiCGSTAB import BiCGSTAB


def load_cuda_imp(Debug=False, Verbose=False):
    src_dir = os.path.dirname(os.path.abspath(__file__)) + "/cuda"
    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(src_dir, "build")
    cflags = ["--extended-lambda", "--expt-relaxed-constexpr"]
    if Debug:
        cflags += ["-G", "-g", "-O0"]
        cflags += ["-DDEBUG"]
    else:
        cflags += ["-O3"]
        cflags += ["-DNDEBUG"]

    cuda_files = glob(src_dir + "/*.cu")
    include_paths = [src_dir + "/include"]
    return load_cuda(
        name="CUDA_MODULE",
        sources=cuda_files,
        extra_include_paths=include_paths,
        extra_cuda_cflags=cflags,
        verbose=Verbose,
    )


cuda_imp = load_cuda_imp(Debug=False, Verbose=True)


def check_tensor(tensor, dtype):
    assert tensor.dtype == dtype
    assert tensor.is_cuda
    assert tensor.is_contiguous()


class BEM_Solver:
    def __init__(self, vertices, triangles):
        check_tensor(vertices, torch.float32)
        check_tensor(triangles, torch.int32)
        self.vertices = vertices
        self.triangles = triangles
        self.device = vertices.device

    def assemble_boundary_matrix(self, wavenumber, layer_type, approx=False):
        return getattr(
            cuda_imp, layer_type + "_boundary_matrix" + ("_approx" if approx else "")
        )(self.vertices, self.triangles, wavenumber)

    def identity_matrix(self):
        return cuda_imp.identity_matrix(self.vertices, self.triangles)


def solve_linear_equation(A_func, b, x=None, nsteps=None, tol=1e-10, atol=1e-16):
    if callable(A_func):
        solver = BiCGSTAB(A_func)
    else:
        solver = BiCGSTAB(lambda x: A_func @ x)
    return solver.solve(b, x=x, nsteps=nsteps, tol=tol, atol=atol)


def get_potential_of_sources(x0, x1, n0, n1, wave_number, degree=0, grad=False):
    check_tensor(x0, torch.float32)
    check_tensor(x1, torch.float32)
    check_tensor(n0, torch.float32)
    check_tensor(n1, torch.float32)
    if degree == 0:
        if grad:
            return cuda_imp.double_boundary_potential(x0, x1, n0, n1, wave_number)
        else:
            return cuda_imp.single_boundary_potential(x0, x1, n0, n1, wave_number)
    elif degree == 1:
        if grad:
            return cuda_imp.hypersingular_boundary_potential(
                x0, x1, n0, n1, wave_number
            )
        else:
            return cuda_imp.adjointdouble_boundary_potential(
                x0, x1, n0, n1, wave_number
            )
    else:
        raise ValueError("degree must be 0 or 1")
