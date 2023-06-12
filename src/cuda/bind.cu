#include <cstdio>
#include "array3D.h"
#include "integrand.h"
#include "macro.h"
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

using namespace GNNBEM;

void __global__ assemble_matrix_kernel(GArr<float3> vertices,
                                       GArr<int3> triangles,
                                       GArr2D<float> matrix,
                                       const float wave_number,
                                       PotentialType type)
{
    for (int i = blockIdx.x; i < triangles.size(); i += gridDim.x)
        for (int j = threadIdx.x; j < triangles.size(); j += blockDim.x)
        {
            matrix(i, j) =
                face2FaceIntegrand(vertices.data(), triangles[j], triangles[i], cpx(wave_number, 0), type).real();
            // if (i == 0 && j == 1)
            // {
            //     printf("i = %d, j = %d, matrix = %e\n", i, j, matrix(i, j));
            //     printf("triangles[i] = %d, %d, %d\n", triangles[i].x, triangles[i].y, triangles[i].z);
            //     printf("triangles[j] = %d, %d, %d\n", triangles[j].x, triangles[j].y, triangles[j].z);
            // }
        }
}

void assemble_matrix(const torch::Tensor &vertices_,
                     const torch::Tensor &triangles_,
                     torch::Tensor &matrix_,
                     const float wave_number,
                     bool is_double_layer)
{
    GArr<float3> vertices((float3 *)vertices_.data_ptr(), vertices_.size(0));
    GArr<int3> triangles((int3 *)triangles_.data_ptr(), triangles_.size(0));
    GArr2D<float> matrix((float *)matrix_.data_ptr(), matrix_.size(0), matrix_.size(1));
    PotentialType type = is_double_layer ? DOUBLE_LAYER : SINGLE_LAYER;
    cuExecuteBlock(triangles.size(), 64, assemble_matrix_kernel, vertices, triangles, matrix, wave_number, type);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("assemble_matrix", &assemble_matrix, "assemble matrix");
}
