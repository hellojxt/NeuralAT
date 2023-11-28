#ifndef LBVH_PREDICATOR_CUH
#define LBVH_PREDICATOR_CUH
#include "aabb.cuh"

namespace lbvh {

template <typename Real, unsigned int dim> struct query_line_intersect {
    using vector_type = typename vector_of<Real, dim>::type;

    __device__ __host__ query_line_intersect(const Line<Real, dim> &line) : line(line) {}

    query_line_intersect() = default;
    ~query_line_intersect() = default;
    query_line_intersect(const query_line_intersect &) = default;
    query_line_intersect(query_line_intersect &&) = default;
    query_line_intersect &operator=(const query_line_intersect &) = default;
    query_line_intersect &operator=(query_line_intersect &&) = default;

    Line<Real, dim> line;
};

template <typename Real, unsigned int dim> struct query_overlap {
    __device__ __host__ query_overlap(const aabb<Real, dim> &tgt) : target(tgt) {}

    query_overlap() = default;
    ~query_overlap() = default;
    query_overlap(const query_overlap &) = default;
    query_overlap(query_overlap &&) = default;
    query_overlap &operator=(const query_overlap &) = default;
    query_overlap &operator=(query_overlap &&) = default;

    __device__ __host__ inline bool operator()(const aabb<Real, dim> &box) noexcept { return intersects(box, target); }

    aabb<Real, dim> target;
};

template <typename Real, unsigned int dim>
__device__ __host__ query_overlap<Real, dim> overlaps(const aabb<Real, dim> &region) noexcept {
    return query_overlap<Real, dim>(region);
}

template <typename Real, unsigned int dim> struct query_nearest {
    using vector_type = typename vector_of<Real, dim>::type;

    __device__ __host__ query_nearest(const vector_type &tgt) : target(tgt) {}

    query_nearest() = default;
    ~query_nearest() = default;
    query_nearest(const query_nearest &) = default;
    query_nearest(query_nearest &&) = default;
    query_nearest &operator=(const query_nearest &) = default;
    query_nearest &operator=(query_nearest &&) = default;

    vector_type target;
};

__device__ __host__ inline query_nearest<float, 2> nearest(const float2 &point) noexcept {
    return query_nearest<float, 2>(point);
}
__device__ __host__ inline query_nearest<double, 2> nearest(const double2 &point) noexcept {
    return query_nearest<double, 2>(point);
}

__device__ __host__ inline query_nearest<float, 3> nearest(const float4 &point) noexcept {
    return query_nearest<float, 3>(point);
}
__device__ __host__ inline query_nearest<float, 3> nearest(const float3 &point) noexcept {
    return query_nearest<float, 3>(make_float4(point.x, point.y, point.z, 0.0f));
}
__device__ __host__ inline query_nearest<double, 3> nearest(const double4 &point) noexcept {
    return query_nearest<double, 3>(point);
}
__device__ __host__ inline query_nearest<double, 3> nearest(const double3 &point) noexcept {
    return query_nearest<double, 3>(make_double4(point.x, point.y, point.z, 0.0));
}

} // namespace lbvh
#endif // LBVH_PREDICATOR_CUH
