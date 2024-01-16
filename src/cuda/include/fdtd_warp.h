#pragma once
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include "common.h"
#include "gpu_memory.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "fdtd.h"
#include "sample.h"
using namespace nwob;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> allocate_grids_data(int res)
{
    torch::Tensor grids = torch::zeros({3, res, res, res}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor pml_grids = torch::zeros({3, res, res, res, sizeof(PMLData) / sizeof(float)},
                                           torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor cells =
        torch::zeros({res, res, res, sizeof(FDTDCell) / sizeof(int)}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    return std::make_tuple(grids, pml_grids, cells);
}

void FDTD_simulation(torch::Tensor vertices_,
                     torch::Tensor triangles_,
                     torch::Tensor triangle_neumann_,
                     torch::Tensor min_bound_,
                     torch::Tensor max_bound_,
                     torch::Tensor grids_,
                     torch::Tensor pml_grids_,
                     torch::Tensor cells_,
                     torch::Tensor signal_,
                     torch::Tensor listen_position_,
                     float dt,
                     int res,
                     int start_time_step)
{
    float3 min_bound =
        make_float3(min_bound_[0].item<float>(), min_bound_[1].item<float>(), min_bound_[2].item<float>());
    float3 max_bound =
        make_float3(max_bound_[0].item<float>(), max_bound_[1].item<float>(), max_bound_[2].item<float>());
    float3 listen_position = make_float3(listen_position_[0].item<float>(), listen_position_[1].item<float>(),
                                         listen_position_[2].item<float>());
    int3 listen_grid_position = make_int3((listen_position - min_bound) / ((max_bound - min_bound) / res));
    int triangles_size = triangles_.size(0);
    // printf("triangles_size: (%d, %d)\n", triangles_size, triangles_.size(1));
    int vertices_size = vertices_.size(0);
    GPUMemory<Element> elements_(triangles_size);
    auto elements = elements_.device_ptr();
    auto vertices = (float3 *)vertices_.data_ptr();
    auto triangles = (int3 *)triangles_.data_ptr();
    parallel_for(triangles_size, [=] __device__(int i) {
        elements[i].v0 = vertices[triangles[i].x];
        elements[i].v1 = vertices[triangles[i].y];
        elements[i].v2 = vertices[triangles[i].z];
        elements[i].id = i;
    });
    // printf("elements init done\n");
    auto bvh = lbvh::bvh<float, 3, Element, ElementAABB>(elements_.begin(), elements_.end());
    // printf("bvh build done\n");
    float grid_size = (max_bound.x - min_bound.x) / res;
    auto states_ = get_random_states(res * res * res);
    // printf("random states done\n");
    FDTDConfig fdtd_config(grid_size, dt, res, min_bound);
    auto states = PitchedPtr<randomState, 3>((randomState *)states_.data_ptr(), res, res, res);
    auto cells = PitchedPtr<FDTDCell, 3>((FDTDCell *)cells_.data_ptr(), res, res, res);
    auto pml_grids = Grids((PMLData *)pml_grids_.data_ptr(), res);
    auto grids = Grids((float *)grids_.data_ptr(), res);
    auto signal = (float *)signal_.data_ptr();
    auto bvh_device = bvh.get_device_repr();
    int simulation_step_num = triangle_neumann_.size(1);
    auto triangle_neumann =
        PitchedPtr<float, 2>((float *)triangle_neumann_.data_ptr(), triangles_size, simulation_step_num);

    parallel_for_3D(res, res, res, [=] __device__(int x, int y, int z) {
        rasterize_obj(cells, bvh_device, fdtd_config, states, x, y, z);
    });
    // printf("rasterize done\n");
    parallel_for_3D(res, res, res, [=] __device__(int x, int y, int z) { mark_ghost_cell(cells, x, y, z); });
    // printf("mark ghost cell done\n");

    for (int t = start_time_step; t < simulation_step_num + start_time_step; t++)
    {
        parallel_for_3D(res, res, res,
                        [=] __device__(int x, int y, int z) { pml_step(fdtd_config, grids, pml_grids, x, y, z, t); });
        // printf("pml step done\n");
        parallel_for_3D(res, res, res, [=] __device__(int x, int y, int z) {
            update_ghost_cell(cells, grids, fdtd_config, elements, triangle_neumann, x, y, z, t);
            if (x == listen_grid_position.x && y == listen_grid_position.y && z == listen_grid_position.z)
            {
                signal[t] = grids[t](listen_grid_position);
            }
        });
    }
    bvh.clear();
}