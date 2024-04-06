#pragma once
#include "gauss.h"
#include "potential.h"

BEM_NAMESPACE_BEGIN
inline HOST_DEVICE int triangle_common_vertex_num(int3 ind1, int3 ind2)
{
    return (ind1.x == ind2.x) + (ind1.x == ind2.y) + (ind1.x == ind2.z) + (ind1.y == ind2.x) + (ind1.y == ind2.y) +
           (ind1.y == ind2.z) + (ind1.z == ind2.x) + (ind1.z == ind2.y) + (ind1.z == ind2.z);
}
// normalized normal of a triangle
inline HOST_DEVICE float3 triangle_norm(float3 *verts)
{
    float3 v1 = verts[1] - verts[0];
    float3 v2 = verts[2] - verts[0];
    float3 n = cross(v1, v2);
    return n / length(n);
}

inline HOST_DEVICE float jacobian(float3 *v)
{
    return length(cross(v[1] - v[0], v[2] - v[0]));
}

inline HOST_DEVICE float jacobian(float3 v1, float3 v2, float3 v3)
{
    return length(cross(v2 - v1, v3 - v1));
}

inline HOST_DEVICE float jacobian(float3 *verts, int3 ind)
{
    return jacobian(verts[ind.x], verts[ind.y], verts[ind.z]);
}

// unit triangle (0, 0), (1, 0), (0, 1)
inline HOST_DEVICE float3 local_to_global(float x1, float x2, float3 *v)
{
    return (1 - x1 - x2) * v[0] + x1 * v[1] + x2 * v[2];
}

// unit triangle (0, 0), (1, 0), (1, 1)
inline HOST_DEVICE float3 local_to_global2(float x1, float x2, float3 *v)
{
    return (1 - x1) * v[0] + (x1 - x2) * v[1] + x2 * v[2];
}

template <PotentialType PType>
inline HOST_DEVICE complex singular_potential(float xsi,
                                              float eta1,
                                              float eta2,
                                              float eta3,
                                              float weight,
                                              float3 *trial_v,
                                              float3 *test_v,
                                              float3 trial_norm,
                                              float3 test_norm,
                                              int neighbor_num,
                                              float s)
{
    complex result = 0;
    xsi = 0.5 * (xsi + 1);
    eta1 = 0.5 * (eta1 + 1);
    eta2 = 0.5 * (eta2 + 1);
    eta3 = 0.5 * (eta3 + 1);
    switch (neighbor_num)
    {
        case 3:
        {  // Indentical Panels
            float w = xsi * xsi * xsi * eta1 * eta1 * eta2;
            float eta12 = eta1 * eta2;
            float eta123 = eta1 * eta2 * eta3;
            float3 v1, v2;
            // Region 1
            v1 = local_to_global2(xsi, xsi * (1.0 - eta1 + eta12), trial_v);
            v2 = local_to_global2(xsi * (1.0 - eta123), xsi * (1.0 - eta1), test_v);
            result += potential<PType>(v1, v2, trial_norm, test_norm, s);
            // Region 2
            v1 = local_to_global2(xsi * (1.0 - eta123), xsi * (1.0 - eta1), trial_v);
            v2 = local_to_global2(xsi, xsi * (1.0 - eta1 + eta12), test_v);
            result += potential<PType>(v1, v2, trial_norm, test_norm, s);
            // Region 3
            v1 = local_to_global2(xsi, xsi * (eta1 - eta12 + eta123), trial_v);
            v2 = local_to_global2(xsi * (1.0 - eta12), xsi * (eta1 - eta12), test_v);
            result += potential<PType>(v1, v2, trial_norm, test_norm, s);
            // Region 4
            v1 = local_to_global2(xsi * (1.0 - eta12), xsi * (eta1 - eta12), trial_v);
            v2 = local_to_global2(xsi, xsi * (eta1 - eta12 + eta123), test_v);
            result += potential<PType>(v1, v2, trial_norm, test_norm, s);
            // Region 5
            v1 = local_to_global2(xsi * (1.0 - eta123), xsi * (eta1 - eta123), trial_v);
            v2 = local_to_global2(xsi, xsi * (eta1 - eta12), test_v);
            result += potential<PType>(v1, v2, trial_norm, test_norm, s);
            // Region 6
            v1 = local_to_global2(xsi, xsi * (eta1 - eta12), trial_v);
            v2 = local_to_global2(xsi * (1.0 - eta123), xsi * (eta1 - eta123), test_v);
            result += potential<PType>(v1, v2, trial_norm, test_norm, s);
            return result * w * weight;
        }
        case 2:
        {  // Common Edge
            float w = xsi * xsi * xsi * eta1 * eta1;
            float eta12 = eta1 * eta2;
            float eta123 = eta1 * eta2 * eta3;
            float3 v1, v2;
            // Region 1
            v1 = local_to_global2(xsi, xsi * eta1 * eta3, trial_v);
            v2 = local_to_global2(xsi * (1.0 - eta12), xsi * eta1 * (1.0 - eta2), test_v);
            result += potential<PType>(v1, v2, trial_norm, test_norm, s);
            // Region 2
            v1 = local_to_global2(xsi, xsi * eta1, trial_v);
            v2 = local_to_global2(xsi * (1.0 - eta123), xsi * eta1 * eta2 * (1 - eta3), test_v);
            result += potential<PType>(v1, v2, trial_norm, test_norm, s) * eta2;
            // Region 3
            v1 = local_to_global2(xsi * (1.0 - eta12), xsi * eta1 * (1.0 - eta2), trial_v);
            v2 = local_to_global2(xsi, xsi * eta123, test_v);
            result += potential<PType>(v1, v2, trial_norm, test_norm, s) * eta2;
            // Region 4
            v1 = local_to_global2(xsi * (1.0 - eta123), xsi * eta12 * (1.0 - eta3), trial_v);
            v2 = local_to_global2(xsi, xsi * eta1, test_v);
            result += potential<PType>(v1, v2, trial_norm, test_norm, s) * eta2;
            // Region 5
            v1 = local_to_global2(xsi * (1.0 - eta123), xsi * eta1 * (1.0 - eta2 * eta3), trial_v);
            v2 = local_to_global2(xsi, xsi * eta12, test_v);
            result += potential<PType>(v1, v2, trial_norm, test_norm, s) * eta2;
            return result * w * weight;
        }
        case 1:
        {  // Common Vertex
            float w = xsi * xsi * xsi * eta2;
            float3 v1, v2;
            // Region 1
            v1 = local_to_global2(xsi, xsi * eta1, trial_v);
            v2 = local_to_global2(xsi * eta2, xsi * eta2 * eta3, test_v);
            result += potential<PType>(v1, v2, trial_norm, test_norm, s);
            // Region 2
            v1 = local_to_global2(xsi * eta2, xsi * eta2 * eta3, trial_v);
            v2 = local_to_global2(xsi, xsi * eta1, test_v);
            result += potential<PType>(v1, v2, trial_norm, test_norm, s);
            return result * w * weight;
        }
    }
}

template <PotentialType PType, int LINE_GAUSS_NUM>
inline HOST_DEVICE complex singular_integrand(float3 *trial_v,
                                              float3 *test_v,
                                              float trial_jacobian,
                                              float test_jacobian,
                                              float s,
                                              int neighbor_num,
                                              float3 trial_norm,
                                              float3 test_norm)
{
    complex result = 0;
    float line_gauss_points[LINE_GAUSS_NUM];
    float line_gauss_weights[LINE_GAUSS_NUM];
    set_line_gauss_params<LINE_GAUSS_NUM>(line_gauss_points, line_gauss_weights);
    for (int xsi_i = 0; xsi_i < LINE_GAUSS_NUM; xsi_i++)
        for (int eta1_i = 0; eta1_i < LINE_GAUSS_NUM; eta1_i++)
            for (int eta2_i = 0; eta2_i < LINE_GAUSS_NUM; eta2_i++)
                for (int eta3_i = 0; eta3_i < LINE_GAUSS_NUM; eta3_i++)
                {
                    result += singular_potential<PType>(line_gauss_points[xsi_i], line_gauss_points[eta1_i],
                                                        line_gauss_points[eta2_i], line_gauss_points[eta3_i],
                                                        line_gauss_weights[xsi_i] * line_gauss_weights[eta1_i] *
                                                            line_gauss_weights[eta2_i] * line_gauss_weights[eta3_i],
                                                        trial_v, test_v, trial_norm, test_norm, neighbor_num, s);
                }
    return result * trial_jacobian * test_jacobian / 16;
}

template <PotentialType PType, int LINE_GAUSS_NUM>
inline HOST_DEVICE complex singular_integrand_thread(float3 *trial_v,
                                                     float3 *test_v,
                                                     float trial_jacobian,
                                                     float test_jacobian,
                                                     float s,
                                                     int neighbor_num,
                                                     float3 trial_norm,
                                                     float3 test_norm,
                                                     int idx)
{
    float line_gauss_points[LINE_GAUSS_NUM];
    float line_gauss_weights[LINE_GAUSS_NUM];
    set_line_gauss_params<LINE_GAUSS_NUM>(line_gauss_points, line_gauss_weights);
    int xsi_i = idx / (LINE_GAUSS_NUM * LINE_GAUSS_NUM * LINE_GAUSS_NUM);
    idx = idx % (LINE_GAUSS_NUM * LINE_GAUSS_NUM * LINE_GAUSS_NUM);
    int eta1_i = idx / (LINE_GAUSS_NUM * LINE_GAUSS_NUM);
    idx = idx % (LINE_GAUSS_NUM * LINE_GAUSS_NUM);
    int eta2_i = idx / LINE_GAUSS_NUM;
    int eta3_i = idx % LINE_GAUSS_NUM;
    return singular_potential<PType>(line_gauss_points[xsi_i], line_gauss_points[eta1_i], line_gauss_points[eta2_i],
                                     line_gauss_points[eta3_i],
                                     line_gauss_weights[xsi_i] * line_gauss_weights[eta1_i] *
                                         line_gauss_weights[eta2_i] * line_gauss_weights[eta3_i],
                                     trial_v, test_v, trial_norm, test_norm, neighbor_num, s) *
           trial_jacobian * test_jacobian / 16;
}

template <PotentialType PType, int TRI_GAUSS_NUM>
inline HOST_DEVICE complex
regular_integrand(float3 *trial_v, float3 *test_v, float trial_jacobian, float test_jacobian, float s)
{
    float tri_gauss_points[TRI_GAUSS_NUM * 2];
    float tri_gauss_weights[TRI_GAUSS_NUM];
    set_tri_gauss_params<TRI_GAUSS_NUM>(tri_gauss_points, tri_gauss_weights);
    complex result = 0;
    float3 trial_norm = triangle_norm(trial_v);
    float3 test_norm = triangle_norm(test_v);

    for (int i = 0; i < TRI_GAUSS_NUM; i++)
        for (int j = 0; j < TRI_GAUSS_NUM; j++)
        {
            float3 v1 = local_to_global(tri_gauss_points[i * 2], tri_gauss_points[i * 2 + 1], trial_v);
            float3 v2 = local_to_global(tri_gauss_points[j * 2], tri_gauss_points[j * 2 + 1], test_v);

            result += 0.25 * tri_gauss_weights[i] * tri_gauss_weights[j] * trial_jacobian * test_jacobian *
                      potential<PType>(v1, v2, trial_norm, test_norm, s);
        }

    return result;
}

template <PotentialType PType, int TRI_GAUSS_NUM>
inline HOST_DEVICE complex potential_integrand(float3 point, float3 *src_v, float src_jacobian, float s)
{
    float tri_gauss_points[TRI_GAUSS_NUM * 2];
    float tri_gauss_weights[TRI_GAUSS_NUM];
    set_tri_gauss_params<TRI_GAUSS_NUM>(tri_gauss_points, tri_gauss_weights);
    complex result = 0;
    float3 src_norm = triangle_norm(src_v);
    float3 trg_norm = {0, 0, 0};

    for (int i = 0; i < TRI_GAUSS_NUM; i++)
    {
        float3 v_in_tri = local_to_global(tri_gauss_points[i * 2], tri_gauss_points[i * 2 + 1], src_v);
        result += 0.5 * tri_gauss_weights[i] * src_jacobian * potential<PType>(v_in_tri, point, src_norm, trg_norm, s);
    }
    return result;
}

template <PotentialType PType, int GAUSS_NUM>
inline HOST_DEVICE complex face2FaceIntegrandRegular(const float3 *vertices, int3 src, int3 trg, float k)
{
    float3 src_v[3] = {{vertices[src.x]}, {vertices[src.y]}, {vertices[src.z]}};
    float src_jacobian = jacobian(src_v);
    float3 trg_v[3] = {{vertices[trg.x]}, {vertices[trg.y]}, {vertices[trg.z]}};
    float trg_jacobian = jacobian(trg_v);
    return regular_integrand<PType, GAUSS_NUM>(src_v, trg_v, src_jacobian, trg_jacobian, k);
}

template <PotentialType PType, int GAUSS_NUM>
inline HOST_DEVICE complex face2FaceIntegrandSingular(const float3 *vertices, int3 src, int3 trg, float k, int idx)
{
    float3 src_v[3] = {{vertices[src.x]}, {vertices[src.y]}, {vertices[src.z]}};
    float src_jacobian = jacobian(src_v);
    float3 trg_v[3] = {{vertices[trg.x]}, {vertices[trg.y]}, {vertices[trg.z]}};
    float trg_jacobian = jacobian(trg_v);
    int neighbor_num = triangle_common_vertex_num(src, trg);
    float3 src_v2[3];
    float3 trg_v2[3];
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

    return singular_integrand_thread<PType, GAUSS_NUM>(src_v2, trg_v2, src_jacobian, trg_jacobian, k, neighbor_num,
                                                       triangle_norm(src_v), triangle_norm(trg_v), idx);
}

template <PotentialType PType, int GAUSS_NUM>
inline HOST_DEVICE complex face2PointIntegrand(const float3 *vertices, int3 src, float3 trg, float k)
{
    float3 src_v[3] = {{vertices[src.x]}, {vertices[src.y]}, {vertices[src.z]}};
    float src_jacobian = jacobian(src_v);
    return potential_integrand<PType, GAUSS_NUM>(trg, src_v, src_jacobian, k);
}

BEM_NAMESPACE_END