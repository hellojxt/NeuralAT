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

template <bool deriv>
void get_monte_carlo_weight(const torch::Tensor trg_points,
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

    parallel_for(N * M, [N, M, k, cdf_sum, trgs = (float3 *)trg_points.data_ptr(),
                         srcs = (float3 *)src_points.data_ptr(), src_normals = (float3 *)src_normals.data_ptr(),
                         out = (float2 *)out.data_ptr(), near_point_num = near_point_num.device_ptr(),
                         src_importance = (float *)src_importance.data_ptr()] __device__(int i) {
        int trg_i = i / M, src_i = i % M;
        float3 trg = trgs[trg_i];
        float3 src = srcs[src_i];
        float3 normal = src_normals[src_i];
        auto weight = Green_func<deriv>(trg, src, normal, k);
        float2 result;
        result.x = weight.real();
        result.y = weight.imag();
        float near_field_cdf_sum = M_PI * EPS * EPS * src_importance[src_i];
        if (length(trg - src) < EPS)
        {
            result *= 2 * (2 * M_PI * EPS) / near_point_num[trg_i];
        }
        else
        {
            result *= 2 * (cdf_sum - near_field_cdf_sum) / src_importance[src_i] / (M - near_point_num[trg_i]);
        }
        out[i] = result;
    });
}

template <bool deriv>
void get_monte_carlo_weight_boundary(const torch::Tensor trg_points,
                                     const torch::Tensor src_points,
                                     const torch::Tensor src_normals,
                                     const torch::Tensor src_importance,
                                     float k,
                                     const float cdf_sum,
                                     torch::Tensor out)
{
    int N = out.size(0), M = out.size(1);
    parallel_for(
        N * M, [N, M, k, cdf_sum, trgs = (float3 *)trg_points.data_ptr(), srcs = (float3 *)src_points.data_ptr(),
                src_normals = (float3 *)src_normals.data_ptr(), out = (float2 *)out.data_ptr(),
                src_importance = (float *)src_importance.data_ptr()] __device__(int i) {
            int trg_i = i / M, src_i = i % M;
            float3 trg = trgs[trg_i];
            float3 src = srcs[src_i];
            float3 normal = src_normals[src_i];
            auto weight = Green_func<deriv>(trg, src, normal, k);
            float2 result;
            result.x = weight.real();
            result.y = weight.imag();
            if (trg_i == src_i)
            {
                result *= 2 * (2 * M_PI * EPS);
            }
            else
            {
                result *= 2 * (cdf_sum - M_PI * EPS * EPS * src_importance[src_i]) / src_importance[src_i] / (M - 1);
            }
            out[i] = result;
        });
}

class Reservoir
{
    public:
        int y_idx;
        float w_sum;
        int M;
        HOST_DEVICE inline Reservoir()
        {
            y_idx = -1;
            w_sum = 0;
            M = 0;
        }

        HOST_DEVICE inline void update(int x_idx, float w_i, float rnd)
        {
            w_sum += w_i;
            M++;
            if (rnd <= w_i / w_sum)
            {
                y_idx = x_idx;
            }
        }
};

template <bool deriv>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_monte_carlo_weight_sparse(const torch::Tensor points_,
                                                                                      const torch::Tensor normals_,
                                                                                      const torch::Tensor importance_,
                                                                                      const torch::Tensor states_,
                                                                                      float k,
                                                                                      const float cdf_sum,
                                                                                      int resample_num)
{
    int N = points_.size(0);
    auto row_indices_ = torch::empty({N + 1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto col_indices_ = torch::empty({N * resample_num}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto values_ = torch::empty({N * resample_num, 2}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto row_indices = (int *)row_indices_.data_ptr();
    auto col_indices = PitchedPtr<int, 2>((int *)col_indices_.data_ptr(), N, resample_num);
    auto values = PitchedPtr<float2, 2>((float2 *)values_.data_ptr(), N, resample_num);
    auto points = (float3 *)points_.data_ptr();
    auto normals = (float3 *)normals_.data_ptr();
    auto importance = (float *)importance_.data_ptr();
    int stride = ceil(N / resample_num);
    auto states = PitchedPtr<randomState, 2>((randomState *)states_.data_ptr(), N, resample_num);
    parallel_for(N * resample_num, [=] __device__(int i) {
        int row_i = i / resample_num, col_i = i % resample_num;
        int trg_i = row_i, start_idx = col_i * stride;
        int end_idx = min(start_idx + stride, N);
        if (col_i > N - (stride - 1) * resample_num)
        {
            start_idx += col_i - (N - (stride - 1) * resample_num);
            end_idx = min(start_idx + stride - 1, N);
        }
        float3 trg = points[trg_i];
        Reservoir reservoir;
        auto state = states(trg_i, col_i);
        float p = 1.0f / (end_idx - start_idx);
        for (int src_i = start_idx; src_i < end_idx; src_i++)
        {
            float p_hat = 1 / (length(trg - points[src_i]) + EPS);
            reservoir.update(src_i, p_hat / p, curand_uniform(&state));
        }
        auto src_i = reservoir.y_idx;
        auto green_func_value = Green_func<deriv>(trg, points[src_i], normals[src_i], k);
        float2 W = make_float2(green_func_value.real(), green_func_value.imag());
        W *= (length(trg - points[src_i]) + EPS) * (1.0f / reservoir.M * reservoir.w_sum);
        if (trg_i == src_i)
        {
            W *= 2 * (2 * M_PI * EPS);
        }
        else
        {
            W *= 2 * (cdf_sum - M_PI * EPS * EPS * importance[src_i]) / importance[src_i] / (N - 1);
        }
        values(trg_i, col_i) = W;
        col_indices(trg_i, col_i) = reservoir.y_idx;
        if (i < N + 1)
            row_indices[i] = i * resample_num;
    });
    return std::make_tuple(row_indices_, col_indices_, values_);
}

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

#define HASH_BUCKET_SIZE 5
#define HASH_TABLE_SIZE_FACTOR 4
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

#include "fdtd.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> allocate_grids_data(int res)
{
    int grid_num = res * res * res;
    torch::Tensor grids = torch::zeros({3, grid_num}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor accumulate_grids = torch::zeros({grid_num}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor pml_grids = torch::zeros({3, grid_num, sizeof(PMLData) / sizeof(float)},
                                           torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor cells =
        torch::zeros({grid_num, sizeof(FDTDCell) / sizeof(int)}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    return std::make_tuple(grids, pml_grids, cells, accumulate_grids);
}

void FDTD_simulation(torch::Tensor vertices_,
                     torch::Tensor triangles_,
                     torch::Tensor triangle_neumann_,
                     torch::Tensor min_bound_,
                     torch::Tensor max_bound_,
                     torch::Tensor grids_,
                     torch::Tensor pml_grids_,
                     torch::Tensor cells_,
                     torch::Tensor accumulate_grids_,
                     float dt,
                     int res,
                     int start_time_step,
                     bool need_rasterize)
{
    float3 min_bound =
        make_float3(min_bound_[0].item<float>(), min_bound_[1].item<float>(), min_bound_[2].item<float>());
    float3 max_bound =
        make_float3(max_bound_[0].item<float>(), max_bound_[1].item<float>(), max_bound_[2].item<float>());
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
    auto accumulate_grids = PitchedPtr<float, 3>((float *)accumulate_grids_.data_ptr(), res, res, res);
    auto bvh_device = bvh.get_device_repr();
    int simulation_step_num = triangle_neumann_.size(1);
    auto triangle_neumann =
        PitchedPtr<float, 2>((float *)triangle_neumann_.data_ptr(), triangles_size, simulation_step_num);

    if (need_rasterize)
    {
        parallel_for_3D(res, res, res, [=] __device__(int x, int y, int z) {
            rasterize_obj(cells, bvh_device, fdtd_config, states, x, y, z);
        });
        // printf("rasterize done\n");
        parallel_for_3D(res, res, res, [=] __device__(int x, int y, int z) { mark_ghost_cell(cells, x, y, z); });
        // printf("mark ghost cell done\n");
    }

    for (int t = start_time_step; t < simulation_step_num + start_time_step; t++)
    {
        parallel_for_3D(res, res, res,
                        [=] __device__(int x, int y, int z) { pml_step(fdtd_config, grids, pml_grids, x, y, z, t); });
        // printf("pml step done\n");
        parallel_for_3D(res, res, res, [=] __device__(int x, int y, int z) {
            update_ghost_cell(cells, grids, accumulate_grids, fdtd_config, elements, triangle_neumann, x, y, z, t);
        });
    }
    bvh.clear();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_random_states", &get_random_states, "");
    m.def("get_cdf", &get_cdf, "");
    m.def("importance_sample", &importance_sample, "");
    m.def("get_monte_carlo_weight1", &get_monte_carlo_weight<true>, "");
    m.def("get_monte_carlo_weight0", &get_monte_carlo_weight<false>, "");
    m.def("get_monte_carlo_weight_boundary1", &get_monte_carlo_weight_boundary<true>, "");
    m.def("get_monte_carlo_weight_boundary0", &get_monte_carlo_weight_boundary<false>, "");
    m.def("get_monte_carlo_weight_sparse1", &get_monte_carlo_weight_sparse<true>, "");
    m.def("get_monte_carlo_weight_sparse0", &get_monte_carlo_weight_sparse<false>, "");
    m.def("poisson_disk_resample", &poisson_disk_resample, "");
    m.def("allocate_grids_data", &allocate_grids_data, "");
    m.def("FDTD_simulation", &FDTD_simulation, "");
}