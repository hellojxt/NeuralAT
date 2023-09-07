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
            cflags += "-G -g"
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


CUDA_MODULE.load(Debug=False, MemoryCheck=False, Verbose=False)


def check_tensor(tensor, dtype):
    assert tensor.dtype == dtype
    assert tensor.is_cuda
    assert tensor.is_contiguous()


class UniformSampler:
    def __init__(self, vertices, triangles, num_samples):
        """
        Initialize a uniform sampler.
        Args:
            vertices: (V, 3) float32 tensor of vertices
            triangles: (T, 3) int32 tensor of vertex indices
            num_samples: number of samples to take
        """
        check_tensor(vertices, torch.float32)
        check_tensor(triangles, torch.int32)
        self.vertices = vertices
        self.triangles = triangles
        self.area_cdf = CUDA_MODULE.get("get_area_cdf")(vertices, triangles)
        self.random_state = CUDA_MODULE.get("get_random_states")(num_samples)
        self.points = torch.zeros(
            (num_samples, 3), dtype=torch.float32, device=self.vertices.device
        )
        self.points_normals = torch.zeros(
            (num_samples, 3), dtype=torch.float32, device=self.vertices.device
        )
        self.inv_pdfs = torch.zeros(
            (num_samples, 1), dtype=torch.float32, device=self.vertices.device
        )
        self.num_samples = num_samples

    def update(self):
        """
        Sample points on the surface of the mesh.
        Returns:
            points: (num_samples, 3) float32 tensor of sampled points
            points_normals: (num_samples, 3) float32 tensor of sampled points normals
            inv_pdfs: (num_samples, 1) float32 tensor of inverse pdfs
        """

        CUDA_MODULE.get("uniform_sample")(
            self.vertices,
            self.triangles,
            self.random_state,
            self.area_cdf,
            self.num_samples,
            self.points,
            self.points_normals,
            self.inv_pdfs,
        )
        return self.points, self.points_normals, self.inv_pdfs


def batch_green_func(trg_points, src_points, src_normals, k, deriv=False):
    """
    Compute the Green function for a batch of target points and a batch of source points.
    Args:
        trg_points: (B, N, 3) float32 tensor of target points
        src_points: (B, M, 3) float32 tensor of source points
        src_normals: (B, M, 3) float32 tensor of source normals
    Returns:
        green_func: (B, N, M) float32 tensor of Green function values
    """
    check_tensor(trg_points, torch.float32)
    check_tensor(src_points, torch.float32)
    check_tensor(src_normals, torch.float32)
    result = torch.zeros(
        (trg_points.shape[0], trg_points.shape[1], src_points.shape[1]),
        dtype=torch.float32,
        device=trg_points.device,
    )
    if deriv:
        CUDA_MODULE.get("batch_green_func_deriv")(
            trg_points, src_points, src_normals, result, k
        )
    else:
        CUDA_MODULE.get("batch_green_func")(
            trg_points, src_points, src_normals, result, k
        )
    return result
