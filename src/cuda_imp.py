from torch.utils.cpp_extension import load as load_cuda
import os
from glob import glob
import torch
import numpy as np


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


def multipole(x0, n0, x, n, k, M, deriv):
    """
    Compute the multipole expansion of the Green's function.
    """
    check_tensor(x0, torch.float32)
    check_tensor(x, torch.float32)
    check_tensor(n, torch.float32)
    cuda_method_name = "get_multipole_values_" + str(M)
    if deriv:
        cuda_method_name += "_deriv"
    return CUDA_MODULE.get(cuda_method_name)(x0, n0, x, n, k)


def check_tensor(tensor, dtype):
    assert tensor.dtype == dtype
    assert tensor.is_cuda
    assert tensor.is_contiguous()


def get_bound_info(vertices, padding=0.1):
    if isinstance(vertices, np.ndarray):
        vertices = torch.from_numpy(vertices)
    min_bound = vertices.min(dim=0)[0]
    max_bound = vertices.max(dim=0)[0]
    bound_size = (max_bound - min_bound).max() * (1 + padding)
    center = (max_bound + min_bound) / 2
    min_bound = center - bound_size / 2
    max_bound = center + bound_size / 2
    return min_bound, max_bound, bound_size


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
        self.min_bound, self.max_bound, self.bound_size = get_bound_info(vertices)

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
        mask = CUDA_MODULE.get("poisson_disk_resample")(
            self.points,
            self.points_normals,
            self.min_bound,
            self.max_bound,
            r,
            k,
        ).bool()
        N = mask.sum()
        self.points = self.points[mask]
        self.points_normals = self.points_normals[mask]
        self.points_importance = self.points_importance[mask]
        self.points_neumann = self.points_neumann[mask]
        self.points_index = self.points_index[mask]
        self.num_samples = N

    def get_points_neumann(self, triangles_neumann):
        return triangles_neumann[self.points_index].to(torch.complex64).reshape(-1, 1)


class MonteCarloWeight:
    def __init__(self, trg_points, src_sample, k=None, deriv=False):
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

    def init_random_states(self, resample_num):
        N = self.src_sample.num_samples
        self.random_state = CUDA_MODULE.get("get_random_states")(N * resample_num)

    def get_weights(self, k=None):
        cuda_method_name = "get_monte_carlo_weight" + str(int(self.deriv))
        CUDA_MODULE.get(cuda_method_name)(
            self.trg_points,
            self.src_sample.points,
            self.src_sample.points_normals,
            self.src_sample.points_importance,
            self.k if k is None else k,
            self.src_sample.cdf[-1],
            self.weights_,
        )
        return torch.view_as_complex(self.weights_)

    def get_weights_potential_ks(self, ks):
        ks = torch.as_tensor(ks, dtype=torch.float32, device=self.trg_points.device)
        cuda_method_name = "get_monte_carlo_weight_potential_ks" + str(int(self.deriv))
        return CUDA_MODULE.get(cuda_method_name)(
            self.trg_points,
            self.src_sample.points,
            self.src_sample.points_normals,
            self.src_sample.points_importance,
            ks,
            self.src_sample.cdf[-1],
        )

    def get_weights_boundary(self, k=None):
        cuda_method_name = "get_monte_carlo_weight_boundary" + str(int(self.deriv))
        CUDA_MODULE.get(cuda_method_name)(
            self.trg_points,
            self.src_sample.points,
            self.src_sample.points_normals,
            self.src_sample.points_importance,
            self.k if k is None else k,
            self.src_sample.cdf[-1],
            self.weights_,
        )
        return torch.view_as_complex(self.weights_)

    def get_weights_boundary_ks(self, ks):
        ks = torch.as_tensor(ks, dtype=torch.float32, device=self.trg_points.device)
        cuda_method_name = "get_monte_carlo_weight_boundary_ks" + str(int(self.deriv))
        return CUDA_MODULE.get(cuda_method_name)(
            self.trg_points,
            self.src_sample.points,
            self.src_sample.points_normals,
            self.src_sample.points_importance,
            ks,
            self.src_sample.cdf[-1],
        )

    def get_weights_sparse(self, resample_num, k=None):
        cuda_method_name = "get_monte_carlo_weight_sparse" + str(int(self.deriv))
        row_indices, col_indices, values = CUDA_MODULE.get(cuda_method_name)(
            self.src_sample.points,
            self.src_sample.points_normals,
            self.src_sample.points_importance,
            self.random_state,
            self.k if k is None else k,
            self.src_sample.cdf[-1],
            resample_num,
        )
        N = len(self.src_sample.points)
        values = torch.view_as_complex(values)
        return torch.sparse_csr_tensor(
            row_indices,
            col_indices,
            values,
            (N, N),
        )

    def get_weights_sparse_ks(self, resample_num, ks):
        ks = torch.as_tensor(ks, dtype=torch.float32, device=self.trg_points.device)
        cuda_method_name = "get_monte_carlo_weight_sparse_ks" + str(int(self.deriv))
        row_indices, col_indices, values = CUDA_MODULE.get(cuda_method_name)(
            self.src_sample.points,
            self.src_sample.points_normals,
            self.src_sample.points_importance,
            self.random_state,
            ks,
            self.src_sample.cdf[-1],
            resample_num,
        )
        N = len(self.src_sample.points)
        M = len(ks)
        row_indices = (
            torch.arange(N * M + 1, device=self.trg_points.device) * resample_num
        )
        col_indices = (
            col_indices
            + torch.arange(M, device=self.trg_points.device).reshape(-1, 1) * N
        ).reshape(-1)
        values = torch.view_as_complex(values).T.reshape(-1)
        return torch.sparse_csr_tensor(
            row_indices,
            col_indices,
            values,
            (N * M, N * M),
        )

    def get_weights_sparse_ks_fast(self, resample_num, ks):
        ks = torch.as_tensor(ks, dtype=torch.float32, device=self.trg_points.device)
        cuda_method_name = "get_monte_carlo_weight_sparse_ks" + str(int(self.deriv))
        row_indices, col_indices, values = CUDA_MODULE.get(cuda_method_name)(
            self.src_sample.points,
            self.src_sample.points_normals,
            self.src_sample.points_importance,
            self.random_state,
            ks,
            self.src_sample.cdf[-1],
            resample_num,
        )
        return col_indices, values


def fast_sparse_matrix_vector_mul(col_indices, values, x):
    return CUDA_MODULE.get("sparse_matrix_vector_mul")(col_indices, values, x)


def fast_sparse_matrix_vector_mul2(col_indices, values, x):
    return CUDA_MODULE.get("sparse_matrix_vector_mul_fast")(col_indices, values, x)


class FDTDSimulator:
    def __init__(self, min_bound, max_bound, bound_size, res):
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.bound_size = bound_size
        self.grid_size = bound_size / res
        self.res = res
        self.dt = self.grid_size / 3**0.5 / 343.0 / 1.01
        self.grids, self.pml_grids, self.cells, self.accumulate_grids = CUDA_MODULE.get(
            "allocate_grids_data"
        )(res)
        self.time_idx = 0

    def get_mgrid_xyz(self):
        x = torch.linspace(
            self.min_bound[0] + self.grid_size / 2,
            self.max_bound[0] - self.grid_size / 2,
            self.res,
            device="cuda",
        )
        y = torch.linspace(
            self.min_bound[1] + self.grid_size / 2,
            self.max_bound[1] - self.grid_size / 2,
            self.res,
            device="cuda",
        )
        z = torch.linspace(
            self.min_bound[2] + self.grid_size / 2,
            self.max_bound[2] - self.grid_size / 2,
            self.res,
            device="cuda",
        )
        return torch.meshgrid(x, y, z, indexing="ij")

    def update(self, vertices, triangles, triangles_neumann, need_rasterize=True):
        CUDA_MODULE.get("FDTD_simulation")(
            vertices,
            triangles,
            triangles_neumann,
            self.min_bound,
            self.max_bound,
            self.grids,
            self.pml_grids,
            self.cells,
            self.accumulate_grids,
            self.dt,
            self.res,
            self.time_idx,
            need_rasterize,
        )
        self.time_idx += triangles_neumann.shape[1]
