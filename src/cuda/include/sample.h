#pragma once
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
                     float e1 = curand_uniform(&random_states[i]);
                     float e2 = curand_uniform(&random_states[i]);
                     float u = 1 - sqrt(e1);
                     float v = e2 * sqrt(e1);
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
