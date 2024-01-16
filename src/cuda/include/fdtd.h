#pragma once
#include "gpu_memory.h"
#include "lbvh.cuh"
#include "intersect.h"
NWOB_NAMESPACE_BEGIN

struct PMLData
{
        float v1[3];
        float v2[3];
};

struct FDTDCell
{
        int is_solid;
        int is_ghost;
        int nearst_face_id;
};

struct FDTDConfig
{
        float c;        // speed of sound
        int res;        // resolution of the grid
        int res_bound;  // resolution of the pml boundary
        float dl;       // grid spacing
        float dt;       // time step
        float damp;
        float3 min_bound;

        FDTDConfig(float dl_, float dt_, int res_, float3 min_bound_)
            : dl(dl_), dt(dt_), res(res_), min_bound(min_bound_)
        {
            c = 343.0f;
            res_bound = 5;
            damp = 2.0f / dl;
        }
};

#define HISTORY 3

template <typename T>
struct Grids
{
        PitchedPtr<T, 3> data[HISTORY];
        HOST_DEVICE Grids(T *ptr, int res)
        {
            for (int i = 0; i < HISTORY; i++)
            {
                data[i].set(ptr, res, res, res);
                ptr += res * res * res;
            }
        }
        HOST_DEVICE inline PitchedPtr<T, 3> &operator[](int i) { return data[(i + HISTORY) % HISTORY]; }
        HOST_DEVICE inline const PitchedPtr<T, 3> &operator[](int i) const { return data[(i + HISTORY) % HISTORY]; }
};

__device__ inline float get_damp(int3 coord, FDTDConfig &fdtd)
{
    if (coord.x < fdtd.res_bound || coord.x >= fdtd.res - fdtd.res_bound || coord.y < fdtd.res_bound ||
        coord.y >= fdtd.res - fdtd.res_bound || coord.z < fdtd.res_bound || coord.z >= fdtd.res - fdtd.res_bound)
        return fdtd.damp;
    else
        return 0;
}

__device__ inline int get_sign(int x, FDTDConfig &fdtd)
{
    if (x > fdtd.res / 2)
        return 1;
    else
        return -1;
}

__device__ inline void
pml_step(FDTDConfig fdtd, Grids<float> grids, Grids<PMLData> pml_grids, int x, int y, int z, int t)
{
    if (x >= fdtd.res - 1 || x == 0 || y >= fdtd.res - 1 || y == 0 || z >= fdtd.res - 1 || z == 0)
        return;
    int3 coord = make_int3(x, y, z);
    float c = fdtd.c;
    float h = fdtd.dl;
    float dt = fdtd.dt;
    int3 e[3] = {make_int3(1, 0, 0), make_int3(0, 1, 0), make_int3(0, 0, 1)};
    int coord_sign[3] = {get_sign(coord.x, fdtd), get_sign(coord.y, fdtd), get_sign(coord.z, fdtd)};
    float c_damp = get_damp(coord, fdtd);
    // grid_value.x = U, grid_value.y = phi, grid_value.z = Phi
    // solve U
    float rhs = 0;
#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        rhs += 1 / (h * h) * (grids[t](coord + e[i]) - 2 * grids[t](coord) + grids[t](coord - e[i]));
        rhs += 1 / h *
               (c_damp * pml_grids[t](coord + coord_sign[i] * e[i]).v2[i] -
                get_damp(coord - coord_sign[i] * e[i], fdtd) * pml_grids[t](coord - coord_sign[i] * e[i]).v1[i]);
    }
    grids[t + 1](coord) = grids[t](coord) * 2 - grids[t - 1](coord) + c * c * dt * dt * rhs;
    if (c_damp == 0)
        return;

#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        float rhs = 0;
        rhs += -1.0f / 2.0f *
               (get_damp(coord - coord_sign[i] * e[i], fdtd) * pml_grids[t](coord - coord_sign[i] * e[i]).v1[i] +
                c_damp * pml_grids[t](coord).v1[i]);
        rhs += -1.0f / (h * 2.0f) * (grids[t](coord + coord_sign[i] * e[i]) - grids[t](coord - coord_sign[i] * e[i]));
        pml_grids[t + 1](coord).v1[i] = pml_grids[t](coord).v1[i] + c * dt * rhs;
    }  // solve phi

#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        float rhs = 0;
        rhs += -1.0f / 2.0f *
               (get_damp(coord - coord_sign[i] * e[i], fdtd) * pml_grids[t](coord).v2[i] +
                c_damp * pml_grids[t](coord + coord_sign[i] * e[i]).v2[i]);
        rhs += -1.0f / (h * 2.0f) * (grids[t](coord + coord_sign[i] * e[i]) - grids[t](coord - coord_sign[i] * e[i]));
        pml_grids[t + 1](coord).v2[i] = pml_grids[t](coord).v2[i] + c * dt * rhs;
    }  // solve Phi
}

__device__ inline void rasterize_obj(PitchedPtr<FDTDCell, 3> cells,
                                     lbvh::bvh_device<float, 3, Element> bvh,
                                     FDTDConfig fdtd,
                                     PitchedPtr<randomState, 3> rand_states,
                                     int x,
                                     int y,
                                     int z)
{
    if (x >= cells.size[0] || y >= cells.size[1] || z >= cells.size[2])
        return;
    int3 coord = make_int3(x, y, z);
    float3 cell_center =
        make_float3(fdtd.min_bound.x + (coord.x + 0.5f) * fdtd.dl, fdtd.min_bound.y + (coord.y + 0.5f) * fdtd.dl,
                    fdtd.min_bound.z + (coord.z + 0.5f) * fdtd.dl);
    const auto nearest = lbvh::query_device(bvh, lbvh::nearest(cell_center), distance_sq_calculator());
    float4 origin = make_float4(cell_center);
    float4 dir;
    int num_odd = 0;
    int num_even = 0;
    for (int test_i = 0; test_i < 16; test_i++)
    {
        float phi = 2.f * M_PI * curand_uniform(&rand_states(coord));
        dir.w = 0.f;
        dir.z = 2.f * curand_uniform(&rand_states(coord)) - 1.f;
        dir.x = sqrtf(1.f - dir.z * dir.z) * cosf(phi);
        dir.y = sqrtf(1.f - dir.z * dir.z) * sinf(phi);
        lbvh::Line<float, 3> line(origin, dir);
        constexpr uint buffer_size = 8;
        thrust::pair<uint, int> buffer[buffer_size];
        auto num_intersections = lbvh::query_device(bvh, lbvh::query_line_intersect<float, 3>(line),
                                                    LineElementIntersect(), buffer, buffer_size);
        if (num_intersections % 2 == 1)
            num_odd++;
        else
            num_even++;
    }
    cells(coord).is_solid = num_odd > num_even ? 1 : 0;
    // cells(coord).is_solid = num_intersections;
    cells(coord).is_ghost = false;
    cells(coord).nearst_face_id = nearest.first;
}

__device__ inline void mark_ghost_cell(PitchedPtr<FDTDCell, 3> cells, int x, int y, int z)
{
    if (x >= cells.size[0] || y >= cells.size[1] || z >= cells.size[2])
        return;
    int3 coord = make_int3(x, y, z);
    if (cells(coord).is_solid == 0)
        return;
    int3 e[6] = {make_int3(1, 0, 0),  make_int3(-1, 0, 0), make_int3(0, 1, 0),
                 make_int3(0, -1, 0), make_int3(0, 0, 1),  make_int3(0, 0, -1)};
    for (int i = 0; i < 6; i++)
    {
        int3 neighbor_coord = coord + e[i];
        if (neighbor_coord.x < 0 || neighbor_coord.x >= cells.size[0] || neighbor_coord.y < 0 ||
            neighbor_coord.y >= cells.size[1] || neighbor_coord.z < 0 || neighbor_coord.z >= cells.size[2])
            continue;
        if (cells(neighbor_coord).is_solid == 0)
            cells(coord).is_ghost = true;
    }
}

__device__ inline void update_ghost_cell(PitchedPtr<FDTDCell, 3> cells,
                                         Grids<float> grids,
                                         FDTDConfig fdtd,
                                         Element *triangles,
                                         PitchedPtr<float, 2> triangles_neumann,
                                         int x,
                                         int y,
                                         int z,
                                         int t)
{
    if (x >= cells.size[0] || y >= cells.size[1] || z >= cells.size[2])
        return;
    int3 coord = make_int3(x, y, z);
    if (cells(coord).is_solid && !cells(coord).is_ghost)
    {
        grids[t + 1](coord) = 0;
        return;
    }
    if (!cells(coord).is_ghost)
        return;
    int3 e[6] = {make_int3(1, 0, 0),  make_int3(-1, 0, 0), make_int3(0, 1, 0),
                 make_int3(0, -1, 0), make_int3(0, 0, 1),  make_int3(0, 0, -1)};
    float value_sum = 0;
    int valid_neighbor_num = 0;
    int neumann_triangle_id = cells(coord).nearst_face_id;
    for (int i = 0; i < 6; i++)
    {
        int3 neighbor_coord = coord + e[i];
        if (neighbor_coord.x < 0 || neighbor_coord.x >= cells.size[0] || neighbor_coord.y < 0 ||
            neighbor_coord.y >= cells.size[1] || neighbor_coord.z < 0 || neighbor_coord.z >= cells.size[2])
            continue;
        if (!cells(neighbor_coord).is_solid)
        {
            float normalized_neumann = triangles_neumann(neumann_triangle_id, t) *
                                       dot(triangles[neumann_triangle_id].normal(), make_float3(e[i]));
            value_sum += grids[t + 1](neighbor_coord) - normalized_neumann * fdtd.dl;
            valid_neighbor_num++;
        }
    }
    if (valid_neighbor_num > 0)
    {
        grids[t + 1](coord) = value_sum / valid_neighbor_num;
        // printf("update ghost cell at (%d, %d, %d, %d) with value %f\n", x, y, z, t, grids[t + 1](coord));
    }
    else
    {
        printf("Detected invalid ghost cell at (%d, %d, %d)\n", x, y, z);
    }
}

NWOB_NAMESPACE_END
