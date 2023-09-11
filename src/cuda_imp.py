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


def check_tensor(tensor, dtype):
    assert tensor.dtype == dtype
    assert tensor.is_cuda
    assert tensor.is_contiguous()


class ImportanceSampler:
    def __init__(self, vertices, triangles, importance, num_samples):
        """
        Initialize a uniform sampler.
        Args:
            vertices: (V, 3) float32 tensor of vertices
            triangles: (T, 3) int32 tensor of vertex indices
            importance: (T, 1) float32 tensor of triangle importances
            num_samples: number of samples to take

        Members:
            points: (num_samples, 3) float32 tensor of sampled points
            points_normals: (num_samples, 3) float32 tensor of sampled points normals
            points_importance: (num_samples, 1) float32 tensor of sampled points importances
        """
        check_tensor(vertices, torch.float32)
        check_tensor(triangles, torch.int32)
        self.vertices = vertices
        self.triangles = triangles
        self.triangle_importance = importance
        self.cdf = CUDA_MODULE.get("get_cdf")(vertices, triangles, importance)
        self.random_state = CUDA_MODULE.get("get_random_states")(num_samples)
        self.points = torch.empty(
            (num_samples, 3), dtype=torch.float32, device=self.vertices.device
        )
        self.points_normals = torch.empty(
            (num_samples, 3), dtype=torch.float32, device=self.vertices.device
        )
        self.points_importance = torch.empty(
            (num_samples), dtype=torch.float32, device=self.vertices.device
        )
        self.num_samples = num_samples

    def update(self):
        """
        Sample points on the surface of the mesh.
        """
        CUDA_MODULE.get("importance_sample")(
            self.vertices,
            self.triangles,
            self.triangle_importance,
            self.cdf,
            self.num_samples,
            self.random_state,
            self.points,
            self.points_normals,
            self.points_importance,
        )


class MonteCarloWeight:
    def __init__(
        self,
        trg_sample: ImportanceSampler,
        src_sample: ImportanceSampler,
        k,
        N=None,
        M=None,
        deriv=False,
    ):
        """
        Compute the Green function for a batch of target points and a batch of source points.
        Args:
            trg_sample: ImportanceSampler for target points
            src_sample: ImportanceSampler for source points
            k: float32 tensor of wave number
            deriv: whether to compute the derivative of the Green function
        Members:
            weights: (num_trg_samples, num_src_samples) float32 tensor
        """
        self.src_sample = src_sample
        self.trg_sample = trg_sample
        if N is None:
            N = trg_sample.num_samples
        if M is None:
            M = src_sample.num_samples
        self.weights_ = torch.empty(
            (N, M),
            dtype=torch.float32,
            device=trg_sample.vertices.device,
        )
        self.k = k
        self.deriv = deriv

    def get_weights(self):
        """
        Compute the Green function for a batch of target points and a batch of source points.
        """
        cuda_method_name = "get_monte_carlo_weight" + str(int(self.deriv))
        CUDA_MODULE.get(cuda_method_name)(
            self.trg_sample.points,
            self.trg_sample.points_importance,
            self.src_sample.points,
            self.src_sample.points_normals,
            self.src_sample.points_importance,
            self.k,
            self.src_sample.cdf[-1],
            self.weights_,
        )
        return self.weights_


def batch_green_func(trg_points, src_points, src_normal, k, deriv=False):
    """
    Compute the Green function for a batch of target points and a batch of source points.
    Args:
        trg_points: (num_trg_samples, 3) float32 tensor of target points
        src_points: (num_src_samples, 3) float32 tensor of source points
        src_normal: (num_src_samples, 3) float32 tensor of source points normals
        k: float32 tensor of wave number
        deriv: whether to compute the derivative of the Green function
    Returns:
        (num_trg_samples, num_src_samples) float32 tensor
    """

    result = torch.empty(
        (trg_points.shape[0], src_points.shape[0]),
        dtype=torch.float32,
        device=trg_points.device,
    )
    cuda_method_name = "batch_green_func" + str(int(deriv))
    CUDA_MODULE.get(cuda_method_name)(
        trg_points,
        src_points,
        src_normal,
        k,
        result,
    )
    return result
