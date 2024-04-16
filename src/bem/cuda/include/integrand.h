#pragma once
#include "common.h"
#include "gauss.h"
#include "potential.h"
#include "gpu_memory.h"

BEM_NAMESPACE_BEGIN
inline __device__ int triangle_common_vertex_num(int3 ind1, int3 ind2)
{
    return (ind1.x == ind2.x) + (ind1.x == ind2.y) + (ind1.x == ind2.z) + (ind1.y == ind2.x) + (ind1.y == ind2.y) +
           (ind1.y == ind2.z) + (ind1.z == ind2.x) + (ind1.z == ind2.y) + (ind1.z == ind2.z);
}
// normalized normal of a triangle
inline __device__ float3 triangle_norm(float3 *verts)
{
    float3 v1 = verts[1] - verts[0];
    float3 v2 = verts[2] - verts[0];
    float3 n = cross(v1, v2);
    return n / length(n);
}

inline __device__ float jacobian(float3 *v)
{
    return length(cross(v[1] - v[0], v[2] - v[0]));
}

inline __device__ float jacobian(float3 v1, float3 v2, float3 v3)
{
    return length(cross(v2 - v1, v3 - v1));
}

inline __device__ float jacobian(float3 *verts, int3 ind)
{
    return jacobian(verts[ind.x], verts[ind.y], verts[ind.z]);
}

// unit triangle (0, 0), (1, 0), (0, 1)
inline __device__ float3 local_to_global(float x1, float x2, float3 *v, float *shape_func_factor)
{
    shape_func_factor[0] = 1 - x1 - x2;
    shape_func_factor[1] = x1;
    shape_func_factor[2] = x2;
    return (1 - x1 - x2) * v[0] + x1 * v[1] + x2 * v[2];
}

// unit triangle (0, 0), (1, 0), (1, 1)
inline __device__ float3 local_to_global2(float x1, float x2, float3 *v, float *shape_func_factor)
{
    shape_func_factor[0] = 1 - x1;
    shape_func_factor[1] = x1 - x2;
    shape_func_factor[2] = x2;
    return (1 - x1) * v[0] + (x1 - x2) * v[1] + x2 * v[2];
}

template <PotentialType PType>
inline __device__ complex singular_potential(float xsi,
                                             float eta1,
                                             float eta2,
                                             float eta3,
                                             float weight,
                                             float3 *trial_v,
                                             float3 *test_v,
                                             float3 trial_norm,
                                             float3 test_norm,
                                             int neighbor_num,
                                             float s,
                                             complex *result,
                                             int thread_idx)
{
    xsi = 0.5 * (xsi + 1);
    eta1 = 0.5 * (eta1 + 1);
    eta2 = 0.5 * (eta2 + 1);
    eta3 = 0.5 * (eta3 + 1);
    // Define variables and lambda function outside the switch statement
    float w;
    float eta12 = eta1 * eta2;
    float eta123 = eta1 * eta2 * eta3;
    float shape_func_factor_trial[3];
    float shape_func_factor_test[3];
    complex log_result = 0;

#ifdef DEBUG
    if (thread_idx == 0)
    {
        printf("singular_potential for thread %d\n", thread_idx);
        printf("trial_v = (%f, %f, %f), (%f, %f, %f), (%f, %f, %f)\n", trial_v[0].x, trial_v[0].y, trial_v[0].z,
               trial_v[1].x, trial_v[1].y, trial_v[1].z, trial_v[2].x, trial_v[2].y, trial_v[2].z);
        printf("test_v = (%f, %f, %f), (%f, %f, %f), (%f, %f, %f)\n", test_v[0].x, test_v[0].y, test_v[0].z,
               test_v[1].x, test_v[1].y, test_v[1].z, test_v[2].x, test_v[2].y, test_v[2].z);
    }

#endif

    auto compute_region = [&](float v1_x, float v1_y, float v2_x, float v2_y, float ww = 1) {
        auto v1 = local_to_global2(v1_x, v1_y, trial_v, shape_func_factor_trial);
        auto v2 = local_to_global2(v2_x, v2_y, test_v, shape_func_factor_test);
        auto local_result = potential<PType>(v1, v2, trial_norm, test_norm, s) * ww;
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                result[ii * 3 + jj] += local_result * shape_func_factor_trial[ii] * shape_func_factor_test[jj];
#ifdef DEBUG
        if (thread_idx == 0)
        {
            printf("thread %d : v1 = (%f, %f, %f), v2 = (%f, %f, %f), local_result = %f + %fj\n", thread_idx, v1.x,
                   v1.y, v1.z, v2.x, v2.y, v2.z, local_result.real(), local_result.imag());
            printf("shape_func_factor_trial = %f, %f, %f\n", shape_func_factor_trial[0], shape_func_factor_trial[1],
                   shape_func_factor_trial[2]);
            printf("shape_func_factor_test = %f, %f, %f\n", shape_func_factor_test[0], shape_func_factor_test[1],
                   shape_func_factor_test[2]);
            printf("weight = %f\n", weight * w);
        }
#endif
    };

    switch (neighbor_num)
    {
        case 3:  // Identical Panels
        {
            w = xsi * xsi * xsi * eta1 * eta1 * eta2;

            // Regions 1-6
            compute_region(xsi, xsi * (1.0 - eta1 + eta12), xsi * (1.0 - eta123), xsi * (1.0 - eta1));
            compute_region(xsi * (1.0 - eta123), xsi * (1.0 - eta1), xsi, xsi * (1.0 - eta1 + eta12));
            compute_region(xsi, xsi * (eta1 - eta12 + eta123), xsi * (1.0 - eta12), xsi * (eta1 - eta12));
            compute_region(xsi * (1.0 - eta12), xsi * (eta1 - eta12), xsi, xsi * (eta1 - eta12 + eta123));
            compute_region(xsi * (1.0 - eta123), xsi * (eta1 - eta123), xsi, xsi * (eta1 - eta12));
            compute_region(xsi, xsi * (eta1 - eta12), xsi * (1.0 - eta123), xsi * (eta1 - eta123));
            break;
        }
        case 2:  // Common Edge
        {
            w = xsi * xsi * xsi * eta1 * eta1;

            // Regions 1-5
            compute_region(xsi, xsi * eta1 * eta3, xsi * (1.0 - eta12), xsi * eta1 * (1.0 - eta2));
            compute_region(xsi, xsi * eta1, xsi * (1.0 - eta123), xsi * eta1 * eta2 * (1 - eta3), eta2);
            compute_region(xsi * (1.0 - eta12), xsi * eta1 * (1.0 - eta2), xsi, xsi * eta123, eta2);
            compute_region(xsi * (1.0 - eta123), xsi * eta12 * (1.0 - eta3), xsi, xsi * eta1, eta2);
            compute_region(xsi * (1.0 - eta123), xsi * eta1 * (1.0 - eta2 * eta3), xsi, xsi * eta12, eta2);
            break;
        }
        case 1:  // Common Vertex
        {
            w = xsi * xsi * xsi * eta2;

            // Regions 1-2
            compute_region(xsi, xsi * eta1, xsi * eta2, xsi * eta2 * eta3);
            compute_region(xsi * eta2, xsi * eta2 * eta3, xsi, xsi * eta1);
            break;
        }
    }
    for (int i = 0; i < 9; i++)
        result[i] *= w * weight;
#ifdef DEBUG
    if (thread_idx == 0)
    {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                printf("result[%d][%d] = %f + %fj\n", i, j, result[i * 3 + j].real(), result[i * 3 + j].imag());
    }
#endif
}

template <PotentialType PType, int LINE_GAUSS_NUM>
inline __device__ void singular_integrand_thread(float3 *trial_v,
                                                 float3 *test_v,
                                                 float trial_jacobian,
                                                 float test_jacobian,
                                                 float s,
                                                 int neighbor_num,
                                                 float3 trial_norm,
                                                 float3 test_norm,
                                                 int thread_idx,
                                                 complex *result)
{
    float line_gauss_points[LINE_GAUSS_NUM];
    float line_gauss_weights[LINE_GAUSS_NUM];
    set_line_gauss_params<LINE_GAUSS_NUM>(line_gauss_points, line_gauss_weights);
    int idx = thread_idx;
    int xsi_i = idx / (LINE_GAUSS_NUM * LINE_GAUSS_NUM * LINE_GAUSS_NUM);
    idx = idx % (LINE_GAUSS_NUM * LINE_GAUSS_NUM * LINE_GAUSS_NUM);
    int eta1_i = idx / (LINE_GAUSS_NUM * LINE_GAUSS_NUM);
    idx = idx % (LINE_GAUSS_NUM * LINE_GAUSS_NUM);
    int eta2_i = idx / LINE_GAUSS_NUM;
    int eta3_i = idx % LINE_GAUSS_NUM;
#ifdef DEBUG
    if (thread_idx == 0)
        printf("singular_integrand_thread for thread %d\n", thread_idx);
#endif
    singular_potential<PType>(line_gauss_points[xsi_i], line_gauss_points[eta1_i], line_gauss_points[eta2_i],
                              line_gauss_points[eta3_i],
                              line_gauss_weights[xsi_i] * line_gauss_weights[eta1_i] * line_gauss_weights[eta2_i] *
                                  line_gauss_weights[eta3_i] * trial_jacobian * test_jacobian / 16,
                              trial_v, test_v, trial_norm, test_norm, neighbor_num, s, result, thread_idx);
}

template <PotentialType PType, int TRI_GAUSS_NUM>
inline __device__ void regular_integrand(float3 *trial_v,
                                         float3 *test_v,
                                         float trial_jacobian,
                                         float test_jacobian,
                                         float s,
                                         complex *result,
                                         bool log)

{
    float tri_gauss_points[TRI_GAUSS_NUM * 2];
    float tri_gauss_weights[TRI_GAUSS_NUM];
    set_tri_gauss_params<TRI_GAUSS_NUM>(tri_gauss_points, tri_gauss_weights);
    float3 trial_norm = triangle_norm(trial_v);
    float3 test_norm = triangle_norm(test_v);
    float shape_func_factor_trial[3];
    float shape_func_factor_test[3];
    for (int i = 0; i < TRI_GAUSS_NUM; i++)
        for (int j = 0; j < TRI_GAUSS_NUM; j++)
        {
            float3 v1 =
                local_to_global(tri_gauss_points[i * 2], tri_gauss_points[i * 2 + 1], trial_v, shape_func_factor_trial);
            float3 v2 =
                local_to_global(tri_gauss_points[j * 2], tri_gauss_points[j * 2 + 1], test_v, shape_func_factor_test);
            complex local_result = 0.25 * tri_gauss_weights[i] * tri_gauss_weights[j] * trial_jacobian * test_jacobian *
                                   potential<PType>(v1, v2, trial_norm, test_norm, s);
#ifdef DEBUG
            if (log)
            {
                printf("(%f + %fj) %f %f\n", local_result.real(), local_result.imag(), shape_func_factor_trial[0],
                       shape_func_factor_test[0]);
            }
#endif
            for (int ii = 0; ii < 3; ii++)
                for (int jj = 0; jj < 3; jj++)
                    result[ii * 3 + jj] += local_result * shape_func_factor_trial[ii] * shape_func_factor_test[jj];
        }
}

// template <PotentialType PType, int TRI_GAUSS_NUM>
// inline __device__ complex potential_integrand(float3 point, float3 *src_v, float src_jacobian, float s)
// {
//     float tri_gauss_points[TRI_GAUSS_NUM * 2];
//     float tri_gauss_weights[TRI_GAUSS_NUM];
//     set_tri_gauss_params<TRI_GAUSS_NUM>(tri_gauss_points, tri_gauss_weights);
//     complex result = 0;
//     float3 src_norm = triangle_norm(src_v);
//     float3 trg_norm = {0, 0, 0};

//     for (int i = 0; i < TRI_GAUSS_NUM; i++)
//     {
//         float3 v_in_tri = local_to_global(tri_gauss_points[i * 2], tri_gauss_points[i * 2 + 1], src_v);
//         result += 0.5 * tri_gauss_weights[i] * src_jacobian * potential<PType>(v_in_tri, point, src_norm, trg_norm,
//         s);
//     }
//     return result;
// }

template <PotentialType PType, int GAUSS_NUM>
inline __device__ void face2FaceIntegrandRegular(const float3 *vertices,
                                                 int3 src,
                                                 int3 trg,
                                                 PitchedPtr<complex, 2> matrix,
                                                 float k,
                                                 bool log)
{
    float3 src_v[3] = {{vertices[src.x]}, {vertices[src.y]}, {vertices[src.z]}};
    float src_jacobian = jacobian(src_v);
    float3 trg_v[3] = {{vertices[trg.x]}, {vertices[trg.y]}, {vertices[trg.z]}};
    float trg_jacobian = jacobian(trg_v);
    complex result[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    regular_integrand<PType, GAUSS_NUM>(src_v, trg_v, src_jacobian, trg_jacobian, k, result, log);
    int src_global_idx[3] = {src.x, src.y, src.z};
    int trg_global_idx[3] = {trg.x, trg.y, trg.z};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            atomicAddCpx(&matrix(src_global_idx[i], trg_global_idx[j]), result[i * 3 + j]);
#ifdef DEBUG
    if (log)
    {
        printf("src = %d %d %d, trg = %d %d %d\n", src.x, src.y, src.z, trg.x, trg.y, trg.z);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                printf("result[%d][%d] = %f + %fi\n", i, j, result[i * 3 + j].real(), result[i * 3 + j].imag());
    }
#endif
}

template <PotentialType PType, int GAUSS_NUM>
inline __device__ void face2FaceIntegrandSingular(const float3 *vertices,
                                                  int3 src,
                                                  int3 trg,
                                                  float k,
                                                  int thread_idx,
                                                  complex *result)
{
    float3 src_v[3] = {{vertices[src.x]}, {vertices[src.y]}, {vertices[src.z]}};
    float src_jacobian = jacobian(src_v);
    float3 trg_v[3] = {{vertices[trg.x]}, {vertices[trg.y]}, {vertices[trg.z]}};
    float trg_jacobian = jacobian(trg_v);
    int neighbor_num = triangle_common_vertex_num(src, trg);
    float3 src_v2[3];
    float3 trg_v2[3];
    complex result_local[9];
    for (int i = 0; i < 9; i++)
        result_local[i] = 0;
    int i[3] = {0, 1, 2};
    int j[3] = {0, 1, 2};
    int src_int[3] = {src.x, src.y, src.z};
    int trg_int[3] = {trg.x, trg.y, trg.z};
    if (neighbor_num == 2)
    {
        int idx = 0;
        for (int jj = 0; jj < 3; jj++)
            for (int ii = 0; ii < 3; ii++)
                if (src_int[ii] == trg_int[jj])
                {
                    i[idx] = ii;
                    j[idx] = jj;
                    idx++;
                }
        i[2] = 3 - i[0] - i[1];
        j[2] = 3 - j[0] - j[1];
    }
    if (neighbor_num == 1)
    {
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                if (src_int[ii] == trg_int[jj])
                {
                    if (ii != 0)
                    {
                        i[0] = ii;
                        i[ii] = 0;
                    }
                    if (jj != 0)
                    {
                        j[0] = jj;
                        j[jj] = 0;
                    }
                }
    }
    for (int idx = 0; idx < 3; idx++)
    {
        src_v2[idx] = src_v[i[idx]];
        trg_v2[idx] = trg_v[j[idx]];
    }
#ifdef DEBUG
    if (thread_idx == 0)
        printf("face2FaceIntegrandSingular for thread %d\n", thread_idx);
#endif
    singular_integrand_thread<PType, GAUSS_NUM>(src_v2, trg_v2, src_jacobian, trg_jacobian, k, neighbor_num,
                                                triangle_norm(src_v), triangle_norm(trg_v), thread_idx, result_local);
    for (int ii = 0; ii < 3; ii++)
        for (int jj = 0; jj < 3; jj++)
            result[i[ii] * 3 + j[jj]] = result_local[ii * 3 + jj];
}

// template <PotentialType PType, int GAUSS_NUM>
// inline __device__ complex face2PointIntegrand(const float3 *vertices, int3 src, float3 trg, float k)
// {
//     float3 src_v[3] = {{vertices[src.x]}, {vertices[src.y]}, {vertices[src.z]}};
//     float src_jacobian = jacobian(src_v);
//     return potential_integrand<PType, GAUSS_NUM>(trg, src_v, src_jacobian, k);
// }

BEM_NAMESPACE_END