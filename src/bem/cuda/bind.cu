#include "common.h"
#include "gpu_memory.h"
#include "integrand.h"
#include "potential.h"
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

using namespace bem;

torch::Tensor identity_matrix(const torch::Tensor &vertices_, const torch::Tensor &triangles_)
{
    int vertices_size = vertices_.size(0);
    int triangles_size = triangles_.size(0);
    torch::Tensor matrix_ =
        torch::zeros({triangles_size, triangles_size}, torch::dtype(torch::kComplexFloat).device(torch::kCUDA));
    PitchedPtr<complex, 2> matrix((complex *)matrix_.data_ptr(), matrix_.size(0), matrix_.size(1));
    float3 *vertices = (float3 *)vertices_.data_ptr();
    int3 *triangles = (int3 *)triangles_.data_ptr();
    parallel_for(triangles_size, [=] __device__(int i) { matrix(i, i) = jacobian(vertices, triangles[i]) * 0.5; });
    return matrix_;
}

template <PotentialType type, int LineGaussNum, int TriGaussNum>
torch::Tensor assemble_matrix(const torch::Tensor &vertices_, const torch::Tensor &triangles_, const float wave_number)
{
    float3 *vertices = (float3 *)vertices_.data_ptr();
    int3 *triangles = (int3 *)triangles_.data_ptr();
    int vertices_size = vertices_.size(0);
    int triangles_size = triangles_.size(0);
    torch::Tensor matrix_ =
        torch::empty({triangles_size, triangles_size}, torch::dtype(torch::kComplexFloat).device(torch::kCUDA));
    PitchedPtr<complex, 2> matrix((complex *)matrix_.data_ptr(), matrix_.size(0), matrix_.size(1));

    parallel_for_block(triangles_size, 512, [=] __device__(int x, int y) {
        int i = x;
        for (int j = y; j < triangles_size; j += blockDim.x)
        {
            if (triangle_common_vertex_num(triangles[i], triangles[j]) == 0)
                matrix(i, j) =
                    face2FaceIntegrandRegular<type, TriGaussNum>(vertices, triangles[i], triangles[j], wave_number);
        }
    });

    parallel_for_block(triangles_size, 256, [=] __device__(int x, int y) {
        int i = x;
        __shared__ int adj[128];
        __shared__ int adj_size;
        // Populate shared memory with triangle adjacency and vertex adjacency
        // information
        if (threadIdx.x == 0)
        {
            adj_size = 0;
        }
        __syncthreads();
        for (int j = y; j < triangles_size; j += blockDim.x)
        {
            int common_vertex_num = triangle_common_vertex_num(triangles[i], triangles[j]);
            if (common_vertex_num > 0)
            {
                adj[atomicAdd_block(&adj_size, 1)] = j;
#ifdef DEBUG
                assert(adj_size < 128);
#endif
            }
        }
        __syncthreads();
        const int sub_integrand_size = LineGaussNum * LineGaussNum * LineGaussNum * LineGaussNum;
        __shared__ complex result[128];
        for (int j = y; j < adj_size; j += blockDim.x)
            result[j] = 0;
        __syncthreads();
        for (int j = y; j < adj_size * sub_integrand_size; j += blockDim.x)
        {
            int j_ = j / sub_integrand_size;   // adj index
            int idx = j % sub_integrand_size;  // sub integrand index
            auto local_result = face2FaceIntegrandSingular<type, LineGaussNum>(vertices, triangles[i],
                                                                               triangles[adj[j_]], wave_number, idx);
            atomicAddCpxBlock(&result[j_], local_result);
        }
        __syncthreads();
        for (int j = y; j < adj_size; j += blockDim.x)
        {
            matrix(i, adj[j]) = result[j];
        }
    });
    return matrix_;
}

template <PotentialType type>
torch::Tensor solve_potential(const torch::Tensor x0_,
                              const torch::Tensor x1_,
                              const torch::Tensor n0_,
                              const torch::Tensor n1_,
                              const float wave_number)
{
    int size0 = x0_.size(0);
    int size1 = x1_.size(0);
    float3 *x0 = (float3 *)x0_.data_ptr();
    float3 *x1 = (float3 *)x1_.data_ptr();
    float3 *n0 = (float3 *)n0_.data_ptr();
    float3 *n1 = (float3 *)n1_.data_ptr();
    torch::Tensor result_ = torch::zeros({size1}, torch::dtype(torch::kComplexFloat).device(torch::kCUDA));
    complex *result = (complex *)result_.data_ptr();
    parallel_for_block(size0, 512, [=] __device__(int i, int j) {
        for (int k = j; k < size1; k += blockDim.x)
        {
            auto local_result = potential<type>(x0[i], x1[k], n0[i], n1[k], wave_number);
            atomicAddCpx(&result[k], local_result);
        }
    });
    return result_;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("single_boundary_matrix", &assemble_matrix<bem::SINGLE_LAYER, 4, 6>,
          "Assemble single layer matrix with 4 and 6 gauss points");
    m.def("single_boundary_matrix_approx", &assemble_matrix<bem::SINGLE_LAYER, 2, 3>,
          "Assemble single layer matrix with 2 and 3 gauss points");
    m.def("single_boundary_matrix_approx_1", &assemble_matrix<bem::SINGLE_LAYER, 2, 1>,
          "Assemble single layer matrix with 1 gauss points");
    m.def("double_boundary_matrix", &assemble_matrix<bem::DOUBLE_LAYER, 4, 6>,
          "Assemble double layer matrix with 4 and 6 gauss points");
    m.def("double_boundary_matrix_approx", &assemble_matrix<bem::DOUBLE_LAYER, 2, 3>,
          "Assemble double layer matrix with 2 and 3 gauss points");
    m.def("double_boundary_matrix_approx_1", &assemble_matrix<bem::DOUBLE_LAYER, 2, 1>,
          "Assemble double layer matrix with 1 gauss points");
    m.def("hypersingular_boundary_matrix", &assemble_matrix<bem::HYPER_SINGLE_LAYER, 4, 6>,
          "Assemble hypersingular layer matrix with 4 and 6 gauss points");
    m.def("hypersingular_boundary_matrix_approx", &assemble_matrix<bem::HYPER_SINGLE_LAYER, 2, 3>,
          "Assemble hypersingular layer matrix with 2 and 3 gauss points");
    m.def("hypersingular_boundary_matrix_approx_1", &assemble_matrix<bem::HYPER_SINGLE_LAYER, 2, 1>,
          "Assemble hypersingular layer matrix with 1 gauss points");
    m.def("adjointdouble_boundary_matrix", &assemble_matrix<bem::ADJOINT_DOUBLE_LAYER, 4, 6>,
          "Assemble adjoint double layer matrix with 4 and 6 gauss points");
    m.def("adjointdouble_boundary_matrix_approx", &assemble_matrix<bem::ADJOINT_DOUBLE_LAYER, 2, 3>,
          "Assemble adjoint double layer matrix with 2 and 3 gauss points");
    m.def("adjointdouble_boundary_matrix_approx_1", &assemble_matrix<bem::ADJOINT_DOUBLE_LAYER, 2, 1>,
          "Assemble adjoint double layer matrix with 1 gauss points");
    m.def("identity_matrix", &identity_matrix, "Assemble identity matrix");
    m.def("single_boundary_potential", &solve_potential<bem::SINGLE_LAYER>, "Solve single layer potential");
    m.def("double_boundary_potential", &solve_potential<bem::DOUBLE_LAYER>, "Solve double layer potential");
    m.def("hypersingular_boundary_potential", &solve_potential<bem::HYPER_SINGLE_LAYER>,
          "Solve hypersingular layer potential");
    m.def("adjointdouble_boundary_potential", &solve_potential<bem::ADJOINT_DOUBLE_LAYER>,
          "Solve adjoint double layer potential");
}