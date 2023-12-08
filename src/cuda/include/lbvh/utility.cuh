#ifndef LBVH_UTILITY_CUH
#define LBVH_UTILITY_CUH
#include <limits>
#ifdef __CUDACC__
#include <math_constants.h>
#include <vector_types.h>
#else
#define __device__
#define __host__
struct uint2 {
    unsigned int x, y;
};
struct uint3 {
    unsigned int x, y, z;
};
struct uint4 {
    unsigned int x, y, z, w;
};
struct float2 {
    float x, y;
};
struct float3 {
    float x, y, z;
};
struct float4 {
    float x, y, z, w;
};
struct double2 {
    double x, y;
};
struct double3 {
    double x, y, z;
};
struct double4 {
    double x, y, z, w;
};
uint2 make_uint2(unsigned int x, unsigned int y) { return {x, y}; }
uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) { return {x, y, z}; }
uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { return {x, y, z, w}; }
float2 make_float2(float x, float y) { return {x, y}; }
float3 make_float3(float x, float y, float z) { return {x, y, z}; }
float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
double2 make_double2(double x, double y) { return {x, y}; }
double3 make_double3(double x, double y, double z) { return {x, y, z}; }
double4 make_double4(double x, double y, double z, double w) { return {x, y, z, w}; }
#endif

namespace lbvh {

template <typename T, unsigned int dim> struct vector_of;
template <> struct vector_of<float, 2> { using type = float2; };
template <> struct vector_of<double, 2> { using type = double2; };
template <> struct vector_of<float, 3> { using type = float4; };
template <> struct vector_of<double, 3> { using type = double4; };

template <typename T, unsigned int dim> using vector_of_t = typename vector_of<T, dim>::type;

template <typename T> __device__ inline T infinity() noexcept;

template <> __device__ inline float infinity<float>() noexcept { return std::numeric_limits<float>::infinity(); }
template <> __device__ inline double infinity<double>() noexcept { return std::numeric_limits<double>::infinity(); }

} // namespace lbvh
#endif // LBVH_UTILITY_CUH
