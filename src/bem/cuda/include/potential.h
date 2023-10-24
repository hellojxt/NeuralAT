#pragma once
#include "array3D.h"
#include "helper_math.h"
#include "macro.h"

namespace GNNBEM
{
enum PotentialType
{
    SINGLE_LAYER,
    DOUBLE_LAYER
};

CGPU_FUNC inline float single_layer_potential(float3 src_coord, float3 trg_coord, float k)
{
    float potential = 0;
    float3 s2t = trg_coord - src_coord;
    float r2 = s2t.x * s2t.x + s2t.y * s2t.y + s2t.z * s2t.z;
    if (r2 != 0)
    {
        float r = sqrt(r2);
        potential += cos(r * k) / (4 * PI * r);
        // printf("r = %f, k = %f, potential = %f + %f i\n", r, k.real(), potential.real(), potential.imag());
    }
    // printf("src_coord = %f, %f, %f, trg_coord = %f, %f, %f, potential = %f + %f i\n", src_coord.x, src_coord.y,
    //        src_coord.z, trg_coord.x, trg_coord.y, trg_coord.z, potential.real(), potential.imag());
    return potential;
}

CGPU_FUNC inline float double_layer_potential(float3 src_coord, float3 trg_coord, float3 trial_norm, float k)
{
    float potential = 0;
    float3 s2t = trg_coord - src_coord;
    float r2 = s2t.x * s2t.x + s2t.y * s2t.y + s2t.z * s2t.z;
    if (r2 != 0)
    {
        float r = sqrt(r2);
        potential += -(cos(r * k) + r * k * sin(r * k)) / (4 * PI * r2 * r) * dot(s2t, trial_norm);
    }
    return potential;
}

CGPU_FUNC inline float layer_potential(float3 src_coord,
                                       float3 trg_coord,
                                       float3 trial_norm,
                                       float k,
                                       PotentialType type)
{
    if (type == SINGLE_LAYER)
        return single_layer_potential(src_coord, trg_coord, k);
    else
        return double_layer_potential(src_coord, trg_coord, trial_norm, k);
}

}  // namespace GNNBEM