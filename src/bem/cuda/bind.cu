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
torch::Tensor assemble_matrix(const torch::Tensor &vertices_,
                              const torch::Tensor &triangles_,
                              const torch::Tensor &normals_,
                              const torch::Tensor &surface_curls_trans_,
                              const float wave_number)
{
    float3 *vertices = (float3 *)vertices_.data_ptr();
    int3 *triangles = (int3 *)triangles_.data_ptr();
    float3 *normals = (float3 *)normals_.data_ptr();
    int vertices_size = vertices_.size(0);
    int triangles_size = triangles_.size(0);
    torch::Tensor matrix_ =
        torch::zeros({vertices_size, vertices_size}, torch::dtype(torch::kComplexFloat).device(torch::kCUDA));
    PitchedPtr<complex, 2> matrix((complex *)matrix_.data_ptr(), matrix_.size(0), matrix_.size(1));
    PitchedPtr<float, 3> surface_curls_trans((float *)surface_curls_trans_.data_ptr(), surface_curls_trans_.size(0),
                                             surface_curls_trans_.size(1), surface_curls_trans_.size(2));

    parallel_for_block(triangles_size, 256, [=] __device__(int x, int y) {
        int i = x;
        for (int j = y; j < triangles_size; j += blockDim.x)
        {
            if (triangle_common_vertex_num(triangles[i], triangles[j]) == 0)
            {
                float curl_product[9];
                matmulABT(&surface_curls_trans(i, 0, 0), &surface_curls_trans(j, 0, 0), curl_product);
                face2FaceIntegrandRegular<type, TriGaussNum>(vertices, dot(normals[i], normals[j]), curl_product,
                                                             triangles[i], triangles[j], matrix, wave_number, true);
            }
        }
    });

    parallel_for_block(triangles_size, 256, [=] __device__(int x, int y) {
        int i = x;
        __shared__ int adj[64];
        __shared__ int adj_size;
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
            }
        }
        __syncthreads();
        const int sub_integrand_size = LineGaussNum * LineGaussNum * LineGaussNum * LineGaussNum;
        __shared__ complex result[64][10];
        __shared__ float normal_prod[64];
        __shared__ float curl_product[64][9];
        for (int j = y; j < adj_size; j += blockDim.x)
        {
            int j_ = adj[j];
            normal_prod[j] = dot(normals[i], normals[j_]);
            matmulABT(&surface_curls_trans(i, 0, 0), &surface_curls_trans(j_, 0, 0), curl_product[j]);
        }
        for (int j = y; j < adj_size * 10; j += blockDim.x)
        {
            result[j / 10][j % 10] = 0;
        }
        __syncthreads();
        for (int j = y; j < adj_size * sub_integrand_size; j += blockDim.x)
        {
            int j_ = j / sub_integrand_size;   // adj index
            int idx = j % sub_integrand_size;  // sub integrand index
            complex local_result[10];
            {
                face2FaceIntegrandSingular<type, LineGaussNum>(vertices, triangles[i], triangles[adj[j_]], wave_number,
                                                               idx, local_result);
                for (int k = 0; k < 10; k++)
                    atomicAddCpxBlock(&result[j_][k], local_result[k]);
            }
        }
        __syncthreads();
        if constexpr (type == HYPER_SINGULAR_LAYER)
        {
            for (int j = y; j < adj_size * 9; j += blockDim.x)
            {
                result[j / 9][j % 9] = curl_product[j / 9][j % 9] * result[j / 9][9] -
                                       result[j / 9][j % 9] * normal_prod[j / 9] * wave_number * wave_number;
            }
            __syncthreads();
        }
        int src_global_idx[3] = {triangles[i].x, triangles[i].y, triangles[i].z};
        for (int j = y; j < adj_size; j += blockDim.x)
        {
            int trg_global_idx[3] = {triangles[adj[j]].x, triangles[adj[j]].y, triangles[adj[j]].z};
            for (int k = 0; k < 3; k++)
                for (int l = 0; l < 3; l++)
                    atomicAddCpx(&matrix(src_global_idx[k], trg_global_idx[l]), result[j][k * 3 + l]);
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
    m.def("hypersingular_boundary_matrix", &assemble_matrix<bem::HYPER_SINGULAR_LAYER, 4, 6>,
          "Assemble hypersingular layer matrix with 4 and 6 gauss points");
    m.def("hypersingular_boundary_matrix_approx", &assemble_matrix<bem::HYPER_SINGULAR_LAYER, 2, 3>,
          "Assemble hypersingular layer matrix with 2 and 3 gauss points");
    m.def("hypersingular_boundary_matrix_approx_1", &assemble_matrix<bem::HYPER_SINGULAR_LAYER, 2, 1>,
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
    m.def("hypersingular_boundary_potential", &solve_potential<bem::HYPER_SINGULAR_LAYER>,
          "Solve hypersingular layer potential");
    m.def("adjointdouble_boundary_potential", &solve_potential<bem::ADJOINT_DOUBLE_LAYER>,
          "Solve adjoint double layer potential");
}