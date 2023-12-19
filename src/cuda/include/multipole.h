#pragma once

#include <cmath>
#include <iostream>
#include "common.h"
#include "helper_math.h"

NWOB_NAMESPACE_BEGIN

// Higher-order multipole sources

template <int M, bool deriv>
HOST_DEVICE inline complex multipole(float3 x, float3 y, float3 xn, float3 yn, float k);

template <>
HOST_DEVICE inline complex multipole<0, false>(float3 x, float3 y, float3 xn, float3 yn, float k)
{
    float r = length(x - y);
    return exp(complex(0, k * r)) / (4 * M_PI * r);
}

template <>
HOST_DEVICE inline complex multipole<0, true>(float3 x, float3 y, float3 xn, float3 yn, float k)
{
    float r = length(x - y);
    complex ikr = complex(0, 1) * r * k;
    complex potential = -exp(ikr) / (4 * M_PI * r * r * r) * (1 - ikr) * dot(y - x, yn);
    return potential;
}

template <>
HOST_DEVICE inline complex multipole<1, false>(float3 x, float3 y, float3 xn, float3 yn, float k)
{
    float r = length(x - y);
    complex ikr = complex(0, 1) * r * k;
    complex potential = -exp(ikr) / (4 * M_PI * r * r * r) * (1 - ikr) * dot(x - y, xn);
    return potential;
}

template <>
HOST_DEVICE inline complex multipole<1, true>(float3 x, float3 y, float3 xn, float3 yn, float k)
{
    float r = length(x - y);
    complex ikr = complex(0, 1) * r * k;
    complex potential =
        exp(ikr) / (4 * M_PI * r * r * r) *
        (-(-3 + 3 * ikr - ikr * ikr) * dot(x - y, xn) * dot(y - x, yn) / (r * r) + (1 - ikr) * dot(xn, yn));
    return potential;
}

NWOB_NAMESPACE_END