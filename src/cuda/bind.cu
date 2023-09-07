#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include "common.h"
#include "gpu_memory.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
using namespace nwob;

torch::Tensor get_random_states(int n)
{
    int state_size = sizeof(randomState);
    torch::Tensor states = torch::empty({n, state_size}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
    GPUMemory<unsigned long long> seeds(n);
    seeds.copy_from_host(get_random_seeds(n));
    parallel_for(n, [states = (randomState *)states.data_ptr(), seeds = seeds.device_ptr()] __device__(int i) {
        curand_init(seeds[i], 0, 0, &states[i]);
    });
    return states;
}

torch::Tensor get_area_cdf(const torch::Tensor vertices, const torch::Tensor triangles)
{
    int vertices_size = vertices.size(0);
    int triangles_size = triangles.size(0);
    torch::Tensor area_cdf = torch::empty({triangles_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    parallel_for(triangles_size, [vertices = (float3 *)vertices.data_ptr(), triangles = (int3 *)triangles.data_ptr(),
                                  area_cdf = (float *)area_cdf.data_ptr()] __device__(int i) {
        float3 v0 = vertices[triangles[i].x];
        float3 v1 = vertices[triangles[i].y];
        float3 v2 = vertices[triangles[i].z];
        area_cdf[i] = length(cross(v1 - v0, v2 - v0)) / 2;
    });
    thrust::inclusive_scan(thrust::device, (float *)area_cdf.data_ptr(), (float *)area_cdf.data_ptr() + triangles_size,
                           (float *)area_cdf.data_ptr(), thrust::plus<float>());
    return area_cdf;
}

void uniform_sample(const torch::Tensor vertices,
                    const torch::Tensor triangles,
                    torch::Tensor random_states,
                    torch::Tensor area_cdf,
                    int num_samples,
                    torch::Tensor points,
                    torch::Tensor points_normals,
                    torch::Tensor inv_pdf)
{
    int vertices_size = vertices.size(0);
    int triangles_size = triangles.size(0);
    parallel_for(num_samples, [vertices = (float3 *)vertices.data_ptr(), triangles = (int3 *)triangles.data_ptr(),
                               area_cdf = (float *)area_cdf.data_ptr(), points = (float3 *)points.data_ptr(),
                               points_normals = (float3 *)points_normals.data_ptr(),
                               random_states = (randomState *)random_states.data_ptr(),
                               inv_pdf = (float *)inv_pdf.data_ptr(), triangles_size] __device__(int i) {
        inv_pdf[i] = area_cdf[triangles_size - 1];
        float x = curand_uniform(&random_states[i]) * area_cdf[triangles_size - 1];
        // binary search
        uint l = 0, r = triangles_size - 1;
        while (l < r)
        {
            uint mid = (l + r) / 2;
            if (area_cdf[mid] < x)
                l = mid + 1;
            else
                r = mid;
        }
        uint element_id = l;
        float u = curand_uniform(&random_states[i]);
        float v = curand_uniform(&random_states[i]);
        if (u + v > 1.f)
        {
            u = 1.f - u;
            v = 1.f - v;
        }
        float w = 1.f - u - v;
        float3 v0 = vertices[triangles[element_id].x];
        float3 v1 = vertices[triangles[element_id].y];
        float3 v2 = vertices[triangles[element_id].z];
        points[i] = v0 * u + v1 * v + v2 * w;
        points_normals[i] = normalize(cross(v1 - v0, v2 - v0));
    });
}

template <int type, bool deriv>
void batch_green_func(const torch::Tensor trg_points,
                      const torch::Tensor src_points,
                      const torch::Tensor src_normals,
                      torch::Tensor out,
                      float k)
{
    int batch_size = trg_points.size(0);
    int N = trg_points.size(1), M = src_points.size(1);
    parallel_for(batch_size * N * M,
                 [N, M, k, trgs = (float3 *)trg_points.data_ptr(), srcs = (float3 *)src_points.data_ptr(),
                  src_normals = (float3 *)src_normals.data_ptr(), out = (float *)out.data_ptr()] __device__(int i) {
                     int batch_id = i / (N * M);
                     int j = i % (N * M);
                     int trg_id = j / M;
                     int src_id = j % M;
                     float3 trg = trgs[batch_id * N + trg_id];
                     float3 src = srcs[batch_id * M + src_id];
                     float3 normal = src_normals[batch_id * M + src_id];
                     out[batch_id * N * M + trg_id * M + src_id] = Green_func<type, deriv>(trg, src, normal, k).real();
                 });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_random_states", &get_random_states, "");
    m.def("get_area_cdf", &get_area_cdf, "");
    m.def("uniform_sample", &uniform_sample, "");
    m.def("batch_green_func_deriv", &batch_green_func<HELMHOLTZ, true>, "");
    m.def("batch_green_func", &batch_green_func<HELMHOLTZ, false>, "");
}