#include <cstdio>
#include "array3D.h"
#include "integrand.h"
#include "macro.h"
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

using namespace GNNBEM;

void __global__ assemble_matrix_regular_kernel(GArr<float3> vertices,
                                               GArr<int3> triangles,
                                               GArr<int> test_indices,
                                               GArr2D<float> matrix,
                                               const float wave_number,
                                               PotentialType type)
{
    for (int i_ = blockIdx.x; i_ < test_indices.size(); i_ += gridDim.x)
    {
        int i = test_indices[i_];
        for (int j = threadIdx.x; j < triangles.size(); j += blockDim.x)
        {
            if (triangle_common_vertex_num(triangles[i], triangles[j]) == 0)
                matrix(i_, j) =
                    face2FaceIntegrandRegular(vertices.data(), triangles[i], triangles[j], wave_number, type);
        }
    }
}

void __global__ assemble_matrix_singular_kernel(GArr<float3> vertices,
                                                GArr<int3> triangles,
                                                GArr<int> test_indices,
                                                GArr2D<float> matrix,
                                                const float wave_number,
                                                PotentialType type)
{
    for (int i_ = blockIdx.x; i_ < test_indices.size(); i_ += gridDim.x)
    {
        int i = test_indices[i_];
        __shared__ int adj[128];
        __shared__ int adj_size;
        // Populate shared memory with triangle adjacency and vertex adjacency information
        if (threadIdx.x == 0)
        {
            adj_size = 0;
        }
        __syncthreads();
        for (int j = threadIdx.x; j < triangles.size(); j += blockDim.x)
        {
            int common_vertex_num = triangle_common_vertex_num(triangles[i], triangles[j]);
            if (common_vertex_num > 0)
            {
                adj[atomicAdd_block(&adj_size, 1)] = j;
#ifdef MEMORY_CHECK
                assert(adj_size < 128);
#endif
            }
        }
        __syncthreads();
        __shared__ float result;
        for (int j = 0; j < adj_size; j++)
        {
            int j_ = adj[j];
            if (threadIdx.x == 0)
            {
                result = 0;
            }
            __syncthreads();
            atomicAdd_block(&result, face2FaceIntegrandSingular256Thread(vertices.data(), triangles[i], triangles[j_],
                                                                         wave_number, type));
            __syncthreads();
            if (threadIdx.x == 0)
            {
                matrix(i_, j_) = result;
            }
        }
    }
}

void assemble_matrix(const torch::Tensor &vertices_,
                     const torch::Tensor &triangles_,
                     const torch::Tensor &test_indices_,
                     torch::Tensor &matrix_,
                     const float wave_number,
                     bool is_double_layer)
{
    GArr<float3> vertices((float3 *)vertices_.data_ptr(), vertices_.size(0));
    GArr<int3> triangles((int3 *)triangles_.data_ptr(), triangles_.size(0));
    GArr<int> test_indices((int *)test_indices_.data_ptr(), test_indices_.size(0));
    GArr2D<float> matrix((float *)matrix_.data_ptr(), matrix_.size(0), matrix_.size(1));
    PotentialType type = is_double_layer ? DOUBLE_LAYER : SINGLE_LAYER;
    cuExecuteBlock(test_indices.size(), 256, assemble_matrix_regular_kernel, vertices, triangles, test_indices, matrix,
                   wave_number, type);
    cuExecuteBlock(test_indices.size(), 256, assemble_matrix_singular_kernel, vertices, triangles, test_indices, matrix,
                   wave_number, type);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("assemble_matrix", &assemble_matrix, "assemble matrix");
}