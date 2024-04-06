#pragma once
#include "common.h"
#include "helper_math.h"

BEM_NAMESPACE_BEGIN

enum PotentialType
{
    SINGLE_LAYER,
    DOUBLE_LAYER,
    ADJOINT_DOUBLE_LAYER,
    HYPER_SINGLE_LAYER,
};
#define EPS 1e-3

template <PotentialType Type>
HOST_DEVICE inline complex potential(float3 x, float3 y, float3 xn, float3 yn, float k)
{
    float r = length(x - y);
    if (r < EPS)
        r = EPS;
    complex ikr = complex(0, 1) * r * k;
    if constexpr (Type == SINGLE_LAYER)
    {
        return exp(ikr) / (4 * M_PI * r);
    }
    else if constexpr (Type == DOUBLE_LAYER)
    {
        return -exp(ikr) / (4 * M_PI * r * r * r) * (1 - ikr) * dot(y - x, yn);
    }
    else if constexpr (Type == ADJOINT_DOUBLE_LAYER)
    {
        return -exp(ikr) / (4 * M_PI * r * r * r) * (1 - ikr) * dot(x - y, xn);
    }
    else if constexpr (Type == HYPER_SINGLE_LAYER)
    {
        return exp(ikr) / (4 * M_PI * r * r * r) *
               ((3 - 3 * ikr - k * k * r * r) * dot(y - x, yn) * dot(x - y, xn) / (r * r) + (1 - ikr) * dot(xn, yn));
    }
}

BEM_NAMESPACE_END