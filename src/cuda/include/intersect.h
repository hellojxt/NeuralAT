#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <thrust/optional.h>
#include <thrust/pair.h>
#include "common.h"
#include "lbvh.cuh"
#include "helper_math.h"
#include <curand_kernel.h>
#include "gpu_memory.h"

NWOB_NAMESPACE_BEGIN

// triangle
struct Element
{
        float3 v0, v1, v2;
        int id;
        inline HOST_DEVICE float3 point(float u, float v) const { return v0 * (1.f - u - v) + v1 * u + v2 * v; }
        inline HOST_DEVICE float3 normal() const { return normalize(cross(v1 - v0, v2 - v0)); }
};

// line element intersection
struct LineElementIntersect
{
        thrust::pair<bool, int> HOST_DEVICE operator()(const lbvh::Line<float, 3> &line,
                                                       const Element &tri) const noexcept
        {
            float3 e1 = tri.v1 - tri.v0;
            float3 e2 = tri.v2 - tri.v0;
            float3 line_dir = make_float3(line.dir);
            float3 line_origin = make_float3(line.origin);
            float3 p = cross(line_dir, e2);
            float det = dot(e1, p);
            if (fabs(det) < 1e-8)
                return thrust::make_pair(false, 0);
            float inv_det = 1.f / det;
            float3 t = line_origin - tri.v0;
            float u = dot(t, p) * inv_det;
            if (u < 0.f || u > 1.f)
                return thrust::make_pair(false, 0);
            float3 q = cross(t, e1);
            float v = dot(line_dir, q) * inv_det;
            if (v < 0.f || u + v > 1.f)
                return thrust::make_pair(false, 0);
            float t_ = dot(e2, q) * inv_det;
            if (t_ < 0.f)
                return thrust::make_pair(false, 0);
            return thrust::make_pair(true, tri.id);
        }
};

// bbox
struct ElementAABB
{
        HOST_DEVICE lbvh::aabb<float, 3> operator()(const Element &tri) const noexcept
        {
            lbvh::aabb<float, 3> aabb;
            aabb.lower = make_float4(fminf(fminf(tri.v0, tri.v1), tri.v2));
            aabb.upper = make_float4(fmaxf(fmaxf(tri.v0, tri.v1), tri.v2));
            return aabb;
        }
};

struct distance_sq_calculator
{
        __device__ float operator()(const float4 pos, const Element &f) const noexcept
        {
            // calculate square distance...
            float3 p = make_float3(pos);
            float3 e1 = f.v1 - f.v0;
            float3 e2 = f.v2 - f.v0;
            float3 t = p - f.v0;
            float3 q = cross(t, e1);
            float3 r = cross(t, e2);
            float3 n = cross(e1, e2);
            float d = dot(n, n);
            float s = dot(q, q);
            float t_ = dot(r, r);
            float3 d1 = make_float3(s, t_, d);
            float3 d2 = make_float3(dot(q, n), dot(r, n), dot(n, n));
            float3 inv_d = make_float3(1.f / d1.x, 1.f / d1.y, 1.f / d1.z);
            float3 b = d2 * inv_d;
            float3 bary = make_float3(1.f - b.x - b.y, b.x, b.y);
            float3 dist = bary * d1 * inv_d;
            return dot(dist, dist);
        }
};

NWOB_NAMESPACE_END