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
        src_dir = os.path.dirname(os.path.abspath(__file__)) + "/cuda"
        os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(src_dir, "build")
        cflags = "--extended-lambda --expt-relaxed-constexpr "
        if Debug:
            cflags += "-G -g -O0"
            cflags += " -DDEBUG"
        else:
            cflags += "-O3"
            cflags += " -DNDEBUG"
        if MemoryCheck:
            cflags += " -DMEMORY_CHECK"
        cuda_files = glob(src_dir + "/*.cu")
        include_paths = [src_dir + "/include"]
        CUDA_MODULE._module = load_cuda(
            name="CUDA_MODULE",
            sources=cuda_files,
            extra_include_paths=include_paths,
            extra_cuda_cflags=[cflags],
            verbose=Verbose,
        )
        return CUDA_MODULE._module


CUDA_MODULE.load(Debug=True, MemoryCheck=False, Verbose=False)


def check_tensor(x, name, type=torch.float32):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.dtype != type:
        raise TypeError(f"{name} must be a {type} tensor")
    if not x.is_cuda:
        raise TypeError(f"{name} must be a CUDA tensor")


class Sampler:
    def __init__(
        self,
        vertices,
        triangles,
        triangle_neumann,
        triangle_importance=None,
        alias_factor=8,
    ):
        check_tensor(vertices, "vertices", torch.float32)
        check_tensor(triangles, "triangles", torch.int32)
        check_tensor(triangle_neumann, "triangle_neumann", torch.float32)
        if triangle_importance is not None:
            check_tensor(triangle_importance, "triangle_importance", torch.float32)
        self.vertices = vertices
        self.triangles = triangles
        self.bbox_min = vertices.min(dim=0)[0]
        self.bbox_max = vertices.max(dim=0)[0]
        self.bbox_max_len = (self.bbox_max - self.bbox_min).max()
        self.triangle_neumann = triangle_neumann
        self.triangle_importance = triangle_importance
        self.alias_factor = alias_factor
        self.sample_table = CUDA_MODULE.get("get_sample_table")(
            self.vertices,
            self.triangles,
            self.triangle_importance,
            self.triangle_neumann,
            self.alias_factor,
        )
        self.point_pairs = CUDA_MODULE.get("allocate_sample_memory")(self.sample_table)
        self.num_samples = self.point_pairs.shape[0]
        self.src_points = torch.empty(
            self.num_samples, 6, dtype=torch.float32, device="cuda"
        )
        self.trg_points = torch.empty(
            self.num_samples, 6, dtype=torch.float32, device="cuda"
        )
        self.A = torch.empty(self.num_samples, 1, dtype=torch.float32, device="cuda")
        self.B = torch.empty(self.num_samples, 1, dtype=torch.float32, device="cuda")
        # dirichlet_trg = A * dirichlet_src + B

    def print_sample_table(self):
        CUDA_MODULE.get("print_alias_table")(self.sample_table)

    def print_point_pairs(self):
        CUDA_MODULE.get("print_point_pairs")(self.point_pairs)

    def sample_points(
        self,
        wave_number,
        reuse_num=0,
        candidate_num=1,
    ):
        CUDA_MODULE.get("sample_alias_table")(
            self.sample_table, self.point_pairs, reuse_num, candidate_num
        )
        CUDA_MODULE.get("get_equation_AB")(
            self.point_pairs,
            self.src_points,
            self.trg_points,
            self.A,
            self.B,
            wave_number,
            self.bbox_min,
            self.bbox_max_len,
        )
