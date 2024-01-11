#pragma once
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include "common.h"
#include "gpu_memory.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
using namespace nwob;
class CellPoint
{
    public:
        float3 pos;
        int origin_point_id;
        long long global_cell_id;
        int3 global_cell_id_3d;
};

class Cell
{
    public:
        long long global_cell_id;  // global cell id
        int3 global_cell_id_3d;
        int first_point_id;          // for sorted points
        int sample_origin_point_id;  // for origin points
        int phase_group_id;
};

#define HASH_BUCKET_SIZE 16
#define HASH_TABLE_SIZE_FACTOR 16
class HashEntry
{
    public:
        int cell_num;
        int cell_ids[HASH_BUCKET_SIZE];
        int global_cell_ids[HASH_BUCKET_SIZE];
};

int inline HOST_DEVICE query_hash_table(const HashEntry *hash_table, int hash_table_size, int global_cell_id)
{
    int hash_idx = global_cell_id % hash_table_size;
    const HashEntry &entry = hash_table[hash_idx];
    int cell_num = entry.cell_num;
    for (int i = 0; i < cell_num; i++)
    {
        if (entry.global_cell_ids[i] == global_cell_id)
            return entry.cell_ids[i];
    }
    return -1;
}

long long inline HOST_DEVICE global_3d_to_1d(int3 global_cell_id_3d, int grid_res)
{
    return global_cell_id_3d.x + global_cell_id_3d.y * grid_res + global_cell_id_3d.z * grid_res * grid_res;
}

int3 inline HOST_DEVICE global_1d_to_3d(long long global_cell_id, int grid_res)
{
    int3 global_cell_id_3d;
    global_cell_id_3d.x = global_cell_id % grid_res;
    global_cell_id_3d.y = (global_cell_id / grid_res) % grid_res;
    global_cell_id_3d.z = global_cell_id / (grid_res * grid_res);
    return global_cell_id_3d;
}

float inline HOST_DEVICE approx_geo_dist(float3 p1, float3 n1, float3 p2, float3 n2)
{
    float de = length(p2 - p1);
    float3 v = (p2 - p1) / de;
    float c1 = dot(n1, v);
    float c2 = dot(n2, v);
    if (abs(c1 - c2) < 1e-3)
        return de / sqrt(1 - c1 * c1);
    return (asin(c1) - asin(c2)) / (c1 - c2) * de;
}

torch::Tensor poisson_disk_resample(const torch::Tensor points,
                                    const torch::Tensor points_normal,
                                    const torch::Tensor min_bound_,
                                    const torch::Tensor max_bound_,
                                    float r,
                                    int k)
{
    float3 min_bound =
        make_float3(min_bound_[0].item<float>(), min_bound_[1].item<float>(), min_bound_[2].item<float>());
    float3 max_bound =
        make_float3(max_bound_[0].item<float>(), max_bound_[1].item<float>(), max_bound_[2].item<float>());
    int dense_point_num = points.size(0);
    float cell_size = r / sqrt(3);
    int grid_res = ceil((max_bound.x - min_bound.x) / cell_size);
    GPUMemory<CellPoint> cell_points(dense_point_num);

    // calculate global cell id
    parallel_for(dense_point_num, [dense_point_num, points = (float3 *)points.data_ptr(), min_bound, cell_size,
                                   grid_res, cell_points = cell_points.device_ptr()] __device__(int i) {
        float3 p = points[i];
        long long x = (p.x - min_bound.x) / cell_size;
        long long y = (p.y - min_bound.y) / cell_size;
        long long z = (p.z - min_bound.z) / cell_size;
        cell_points[i].global_cell_id_3d = make_int3(x, y, z);
        cell_points[i].global_cell_id = global_3d_to_1d(make_int3(x, y, z), grid_res);
        cell_points[i].pos = p;
        cell_points[i].origin_point_id = i;
    });

    // sort by global cell id
    thrust::sort(thrust::device, cell_points.begin(), cell_points.end(),
                 [] __device__(const CellPoint &a, const CellPoint &b) { return a.global_cell_id < b.global_cell_id; });

    // calculate the valid cells and the phase group id
    GPUMemory<Cell> cells(dense_point_num);
    parallel_for(dense_point_num, [dense_point_num, cell_points = cell_points.device_ptr(), grid_res,
                                   cells = cells.device_ptr()] __device__(int i) {
        if (i == 0 || cell_points[i].global_cell_id != cell_points[i - 1].global_cell_id)
        {
            int global_cell_id = cell_points[i].global_cell_id;
            int3 global_cell_id_3d = cell_points[i].global_cell_id_3d;
            cells[i].global_cell_id = global_cell_id;
            cells[i].first_point_id = i;
            cells[i].sample_origin_point_id = -1;
            cells[i].global_cell_id_3d = global_cell_id_3d;
            long long x = global_cell_id_3d.x;
            long long y = global_cell_id_3d.y;
            long long z = global_cell_id_3d.z;
            cells[i].phase_group_id = (x % 3) + (y % 3) * 3 + (z % 3) * 3 * 3;
        }
        else
        {
            cells[i].global_cell_id = -1;
        }
    });

    // print cell info
    // parallel_for(1, [cells = cells.device_ptr(), points = (float3 *)points.data_ptr()] __device__(int i) {
    //     for (int j = 0; j < 10; j++)
    //     {
    //         printf(
    //             "cell %d: global_cell_id: %ld, global_cell_id_3d: %d %d %d, phase_group_id: %d, "
    //             "first_point_id: %d, first point: %f %f %f\n",
    //             j, cells[j].global_cell_id, cells[j].global_cell_id_3d.x, cells[j].global_cell_id_3d.y,
    //             cells[j].global_cell_id_3d.z, cells[j].phase_group_id, cells[j].first_point_id,
    //             points[cells[j].first_point_id].x, points[cells[j].first_point_id].y,
    //             points[cells[j].first_point_id].z);
    //     }
    // });

    Cell *new_cell_end = thrust::remove_if(thrust::device, cells.begin(), cells.end(),
                                           [] __device__(const Cell &c) { return c.global_cell_id == -1; });
    int valid_cell_num = new_cell_end - cells.begin();

    // print cell info
    // parallel_for(1,
    //              [valid_cell_num, cells = cells.device_ptr(), points = (float3 *)points.data_ptr()]
    //              __device__(int i)
    //              {
    //                  for (int j = 0; j < valid_cell_num; j++)
    //                  {
    //                      printf(
    //                          "cell %d: global_cell_id: %ld, global_cell_id_3d: %d %d %d, phase_group_id: %d, "
    //                          "first_point_id: %d, first point: %f %f %f\n",
    //                          j, cells[j].global_cell_id, cells[j].global_cell_id_3d.x,
    //                          cells[j].global_cell_id_3d.y, cells[j].global_cell_id_3d.z, cells[j].phase_group_id,
    //                          cells[j].first_point_id, points[cells[j].first_point_id].x,
    //                          points[cells[j].first_point_id].y, points[cells[j].first_point_id].z);
    //                  }
    //              });

    // sort by phase group id
    thrust::sort(thrust::device, cells.begin(), cells.begin() + valid_cell_num,
                 [] __device__(const Cell &a, const Cell &b) { return a.phase_group_id < b.phase_group_id; });

    // initialize the hash table
    int hash_table_size = valid_cell_num * HASH_TABLE_SIZE_FACTOR;
    GPUMemory<HashEntry> hash_table(hash_table_size);
    hash_table.memset(0);
    parallel_for(valid_cell_num, [valid_cell_num, cells = cells.device_ptr(), hash_table = hash_table.device_ptr(),
                                  hash_table_size] __device__(int i) {
        int hash_idx = cells[i].global_cell_id % hash_table_size;
        HashEntry &entry = hash_table[hash_idx];
        int cell_idx_in_bucket = atomicAdd(&entry.cell_num, 1);
        entry.cell_ids[cell_idx_in_bucket] = i;
        entry.global_cell_ids[cell_idx_in_bucket] = cells[i].global_cell_id;
    });

    // start sampling
    for (int trial_t = 0; trial_t < k; trial_t++)
    {
        // get random group id list
        int random_group_num = 3 * 3 * 3;
        torch::Tensor random_group_ids =
            torch::randperm(random_group_num, torch::dtype(torch::kInt32).device(torch::kCUDA));
        // for each random group
        for (int phase = 0; phase < random_group_num; phase++)
        {
            int phase_group_id = random_group_ids[phase].item<int>();
            // int phase_group_id = phase;
            // printf("phase_group_id: %d\n", phase_group_id);
            parallel_for(valid_cell_num, [valid_cell_num, phase_group_id, cells = cells.device_ptr(), trial_t, r,
                                          dense_point_num, points = (float3 *)points.data_ptr(),
                                          points_normal = (float3 *)points_normal.data_ptr(),
                                          cell_points = cell_points.device_ptr(), grid_res,
                                          hash_table = hash_table.device_ptr(), hash_table_size] __device__(int i) {
                auto &cell = cells[i];
                if (cell.phase_group_id == phase_group_id && cell.first_point_id + trial_t < dense_point_num)
                {
                    CellPoint &p = cell_points[cell.first_point_id + trial_t];
                    if (p.global_cell_id != cell.global_cell_id)
                        return;
                    bool conflict = false;
                    int3 global_cell_id_3d = cell.global_cell_id_3d;
                    for (int x = -2; x <= 2; x++)
                    {
                        for (int y = -2; y <= 2; y++)
                        {
                            for (int z = -2; z <= 2; z++)
                            {
                                int3 neighbor_global_cell_id_3d = global_cell_id_3d + make_int3(x, y, z);
                                if (neighbor_global_cell_id_3d.x < 0 || neighbor_global_cell_id_3d.x >= grid_res ||
                                    neighbor_global_cell_id_3d.y < 0 || neighbor_global_cell_id_3d.y >= grid_res ||
                                    neighbor_global_cell_id_3d.z < 0 || neighbor_global_cell_id_3d.z >= grid_res)
                                    continue;
                                int neighbor_global_cell_id = global_3d_to_1d(neighbor_global_cell_id_3d, grid_res);
                                int neighbor_cell_id =
                                    query_hash_table(hash_table, hash_table_size, neighbor_global_cell_id);
                                // printf("%d:neighbor_global_cell_id: 1d: %d, 3d: %d %d %d, neighbor_cell_id:
                                // %d\n",
                                //        phase_group_id, neighbor_global_cell_id, neighbor_global_cell_id_3d.x,
                                //        neighbor_global_cell_id_3d.y, neighbor_global_cell_id_3d.z,
                                //        neighbor_cell_id);
                                if (neighbor_cell_id == -1)
                                    continue;
                                int neighbor_p_id = cells[neighbor_cell_id].sample_origin_point_id;
                                // printf("%d:neighbor_cell_id: %d, neighbor_p_id: %d\n", phase_group_id,
                                // neighbor_cell_id,
                                // neighbor_p_id);
                                if (neighbor_p_id == -1)
                                    continue;
                                float3 neighbor_p = points[neighbor_p_id];
                                float3 neighbor_p_normal = points_normal[neighbor_p_id];
                                float3 current_p = points[p.origin_point_id];
                                float3 current_p_normal = points_normal[p.origin_point_id];
                                // printf("%d: p: %f %f %f, neighbor_p: %f %f %f\n", phase_group_id, p.pos.x,
                                // p.pos.y,
                                // p.pos.z, neighbor_p.x, neighbor_p.y, neighbor_p.z);
                                if (approx_geo_dist(current_p, current_p_normal, neighbor_p, neighbor_p_normal) < r)
                                {
                                    conflict = true;
                                    break;
                                }
                            }
                            if (conflict)
                                break;
                        }
                        if (conflict)
                            break;
                    }
                    if (!conflict)
                    {
                        cell.sample_origin_point_id = p.origin_point_id;
                        // printf("%d: sampled of cell %ld (%d, %d, %d): %f %f %f\n", phase_group_id,
                        // cell.global_cell_id,
                        //        cell.global_cell_id_3d.x, cell.global_cell_id_3d.y, cell.global_cell_id_3d.z,
                        //        p.pos.x, p.pos.y, p.pos.z);
                    }
                }
            });
        }
    }

    // get the valid points mask
    torch::Tensor valid_points_mask = torch::zeros({dense_point_num}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    parallel_for(valid_cell_num, [valid_cell_num, cells = cells.device_ptr(),
                                  valid_points_mask = (int *)valid_points_mask.data_ptr()] __device__(int i) {
        int origin_point_id = cells[i].sample_origin_point_id;
        if (origin_point_id != -1)
            valid_points_mask[origin_point_id] = 1;
    });

    return valid_points_mask;
}
