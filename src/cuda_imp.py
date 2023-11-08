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


CUDA_MODULE.load(Debug=False, MemoryCheck=False, Verbose=False)


def check_tensor(tensor, dtype):
    assert tensor.dtype == dtype
    assert tensor.is_cuda
    assert tensor.is_contiguous()


class ImportanceSampler:
    def __init__(
        self, vertices, triangles, importance, num_samples, neumann_coeff=None
    ):
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
        if neumann_coeff is None:
            neumann_coeff = torch.ones(
                len(self.triangles), dtype=torch.float32, device=self.vertices.device
            )
        self.triangle_neumann = neumann_coeff.float().reshape(-1, 1)
        self.points_neumann = torch.empty(
            (num_samples, 1), dtype=torch.float32, device=self.vertices.device
        )
        self.points_index = torch.empty(
            (num_samples), dtype=torch.int32, device=self.vertices.device
        )

    def update(self):
        """
        Sample points on the surface of the mesh.
        """
        CUDA_MODULE.get("importance_sample")(
            self.vertices,
            self.triangles,
            self.triangle_importance,
            self.triangle_neumann,
            self.cdf,
            self.num_samples,
            self.random_state,
            self.points,
            self.points_normals,
            self.points_importance,
            self.points_neumann,
            self.points_index,
        )

    def poisson_disk_resample(self, r, k=5):
        """
        Sample points on the surface of the mesh.
        """
        min_bound = self.vertices.min(dim=0)[0]
        max_bound = self.vertices.max(dim=0)[0]
        bound_size = (max_bound - min_bound).max()
        center = (max_bound + min_bound) / 2
        min_bound = center - bound_size / 2 * 1.1
        max_bound = center + bound_size / 2 * 1.1
        return CUDA_MODULE.get("poisson_disk_resample")(
            self.points,
            self.points_normals,
            min_bound,
            max_bound,
            r,
            k,
        )


class MonteCarloWeight:
    def __init__(self, trg_points, src_sample, k, deriv=False):
        self.src_sample = src_sample
        self.trg_points = trg_points
        N = trg_points.shape[0]
        M = src_sample.num_samples
        self.weights_ = torch.empty(
            (N, M, 2),
            dtype=torch.float32,
            device=trg_points.device,
        )
        self.k = k
        self.deriv = deriv

    def get_weights(self):
        """
        Compute the Green function for a batch of target points and a batch of source points.
        """
        cuda_method_name = "get_monte_carlo_weight" + str(int(self.deriv))
        CUDA_MODULE.get(cuda_method_name)(
            self.trg_points,
            self.src_sample.points,
            self.src_sample.points_normals,
            self.src_sample.points_importance,
            self.k,
            self.src_sample.cdf[-1],
            self.weights_,
        )
        return torch.view_as_complex(self.weights_)
