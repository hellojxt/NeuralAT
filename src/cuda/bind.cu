#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include "common.h"
#include "gpu_memory.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
using namespace nwob;

void uniform_sample(const torch::Tensor vertices,
                    const torch::Tensor triangles,
                    int num_samples,
                    torch::Tensor points,
                    torch::Tensor points_normals)
{
    GPUMemory<unsigned long long> seeds(num_samples);
    seeds.copy_from_host(get_random_seeds(num_samples));
    int vertices_size = vertices.size(0);
    int triangles_size = triangles.size(0);
    GPUMemory<float> area_cdf(triangles_size);
    parallel_for(triangles_size, [vertices = (float3 *)vertices.data_ptr(), triangles = (int3 *)triangles.data_ptr(),
                                  area_cdf = area_cdf.device_ptr()] __device__(int i) {
        float3 v0 = vertices[triangles[i].x];
        float3 v1 = vertices[triangles[i].y];
        float3 v2 = vertices[triangles[i].z];
        area_cdf[i] = length(cross(v1 - v0, v2 - v0)) / 2;
    });
    thrust::inclusive_scan(thrust::device, area_cdf.begin(), area_cdf.end(), area_cdf.begin(), thrust::plus<float>());
    parallel_for(num_samples, [vertices = (float3 *)vertices.data_ptr(), triangles = (int3 *)triangles.data_ptr(),
                               area_cdf = area_cdf.device_ptr(), points = (float3 *)points.data_ptr(),
                               points_normals = (float3 *)points_normals.data_ptr(), seeds = seeds.device_ptr(),
                               triangles_size] __device__(int i) {
        randomState rand_state;
        curand_init(seeds[i], 0, 0, &rand_state);
        float x = curand_uniform(&rand_state) * area_cdf[triangles_size - 1];
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
        float u = curand_uniform(&rand_state);
        float v = curand_uniform(&rand_state);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(STR(uniform_sample), &uniform_sample, "");
}