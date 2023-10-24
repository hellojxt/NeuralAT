#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include "common.h"
#include "gpu_memory.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
using namespace nwob;

#define EPS 1e-3
template <bool deriv>
HOST_DEVICE inline complex Green_func(float3 y, float3 x, float3 xn, float k);

template <>
HOST_DEVICE inline complex Green_func<false>(float3 y, float3 x, float3 xn, float k)
{
    float r = length(x - y);
    if (r < EPS)
        return exp(complex(0, k * r)) / (4 * M_PI);
    return exp(complex(0, k * r)) / (4 * M_PI * r);
}

template <>
HOST_DEVICE inline complex Green_func<true>(float3 y, float3 x, float3 xn, float k)
{
    float r = length(x - y);
    if (r < EPS)
        return 0;
    complex ikr = complex(0, 1) * r * k;
    complex potential = -exp(ikr) / (4 * M_PI * r * r * r) * (1 - ikr) * dot(x - y, xn);
    return potential;
}

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

torch::Tensor get_cdf(const torch::Tensor vertices,
                      const torch::Tensor triangles,
                      const torch::Tensor triangle_importance)
{
    int vertices_size = vertices.size(0);
    int triangles_size = triangles.size(0);
    torch::Tensor cdf = torch::empty({triangles_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    parallel_for(triangles_size, [vertices = (float3 *)vertices.data_ptr(), triangles = (int3 *)triangles.data_ptr(),
                                  cdf = (float *)cdf.data_ptr(),
                                  triangle_importance = (float *)triangle_importance.data_ptr()] __device__(int i) {
        float3 v0 = vertices[triangles[i].x];
        float3 v1 = vertices[triangles[i].y];
        float3 v2 = vertices[triangles[i].z];
        cdf[i] = length(cross(v1 - v0, v2 - v0)) / 2 * triangle_importance[i];
    });

    thrust::inclusive_scan(thrust::device, (float *)cdf.data_ptr(), (float *)cdf.data_ptr() + triangles_size,
                           (float *)cdf.data_ptr(), thrust::plus<float>());
    return cdf;
}

void importance_sample(const torch::Tensor vertices,
                       const torch::Tensor triangles,
                       torch::Tensor triangles_importance,
                       torch::Tensor triangles_neumann,
                       torch::Tensor cdf,
                       int num_samples,
                       torch::Tensor random_states,
                       torch::Tensor points,
                       torch::Tensor points_normals,
                       torch::Tensor points_importance,
                       torch::Tensor points_neumann,
                       torch::Tensor points_index)

{
    int vertices_size = vertices.size(0);
    int triangles_size = triangles.size(0);
    parallel_for(num_samples,
                 [vertices = (float3 *)vertices.data_ptr(), triangles = (int3 *)triangles.data_ptr(),
                  cdf = (float *)cdf.data_ptr(), points = (float3 *)points.data_ptr(),
                  triangles_importance = (float *)triangles_importance.data_ptr(),
                  triangles_neumann = (float *)triangles_neumann.data_ptr(),
                  points_normals = (float3 *)points_normals.data_ptr(),
                  points_importance = (float *)points_importance.data_ptr(),
                  points_neumann = (float *)points_neumann.data_ptr(), points_index = (int *)points_index.data_ptr(),
                  random_states = (randomState *)random_states.data_ptr(), triangles_size] __device__(int i) {
                     float x = curand_uniform(&random_states[i]) * cdf[triangles_size - 1];
                     // binary search
                     uint l = 0, r = triangles_size - 1;
                     while (l < r)
                     {
                         uint mid = (l + r) / 2;
                         if (cdf[mid] < x)
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
                     points_importance[i] = triangles_importance[element_id];
                     points_neumann[i] = triangles_neumann[element_id];
                     points_index[i] = element_id;
                 });
}

template <bool deriv>
void get_monte_carlo_weight(const torch::Tensor trg_points,
                            const torch::Tensor trg_importance,
                            const torch::Tensor src_points,
                            const torch::Tensor src_normals,
                            const torch::Tensor src_importance,
                            float k,
                            const float cdf_sum,
                            torch::Tensor out)
{
    int N = out.size(0), M = out.size(1);
    GPUMemory<int> near_point_num(N);
    near_point_num.memset(0);
    parallel_for(N * M, [N, M, k, trgs = (float3 *)trg_points.data_ptr(), srcs = (float3 *)src_points.data_ptr(),
                         near_point_num = near_point_num.device_ptr()] __device__(int i) {
        if (length(trgs[i / M] - srcs[i % M]) < EPS)
            atomicAdd(&near_point_num[i / M], 1);
    });

    parallel_for(
        N * M, [N, M, k, cdf_sum, trgs = (float3 *)trg_points.data_ptr(), srcs = (float3 *)src_points.data_ptr(),
                src_normals = (float3 *)src_normals.data_ptr(), out = (float *)out.data_ptr(),
                near_point_num = near_point_num.device_ptr(), src_importance = (float *)src_importance.data_ptr(),
                trg_importance = (float *)trg_importance.data_ptr()] __device__(int i) {
            int trg_i = i / M, src_i = i % M;
            float3 trg = trgs[trg_i];
            float3 src = srcs[src_i];
            float3 normal = src_normals[src_i];
            out[i] = Green_func<deriv>(trg, src, normal, k).real();
            float near_field_cdf_sum = M_PI * EPS * EPS * src_importance[src_i];
            if (length(trg - src) < EPS)
            {
                out[i] *= 2 * (2 * M_PI * EPS) / near_point_num[trg_i];
            }
            else
            {
                out[i] *= 2 * (cdf_sum - near_field_cdf_sum) / src_importance[src_i] / (M - near_point_num[trg_i]);
            }
        });
}

template <bool deriv>
void batch_green_func(const torch::Tensor trg_points,
                      const torch::Tensor src_points,
                      const torch::Tensor src_normals,
                      float k,
                      torch::Tensor out)
{
    int N = trg_points.size(0), M = src_points.size(0);
    parallel_for(N * M,
                 [N, M, k, trgs = (float3 *)trg_points.data_ptr(), srcs = (float3 *)src_points.data_ptr(),
                  src_normals = (float3 *)src_normals.data_ptr(), out = (float *)out.data_ptr()] __device__(int i) {
                     int trg_i = i / M, src_i = i % M;
                     float3 trg = trgs[trg_i];
                     float3 src = srcs[src_i];
                     float3 normal = src_normals[src_i];
                     out[i] = Green_func<deriv>(trg, src, normal, k).real();
                 });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_random_states", &get_random_states, "");
    m.def("get_cdf", &get_cdf, "");
    m.def("importance_sample", &importance_sample, "");
    m.def("get_monte_carlo_weight1", &get_monte_carlo_weight<true>, "");
    m.def("get_monte_carlo_weight0", &get_monte_carlo_weight<false>, "");
    m.def("batch_green_func1", &batch_green_func<true>, "");
    m.def("batch_green_func0", &batch_green_func<false>, "");
}