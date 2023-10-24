#pragma once
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_scan.h>
#include <bitset>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>

namespace GNNBEM
{
#define LOG_ERROR_COLOR "\033[1;31m"
#define LOG_WARNING_COLOR "\033[1;33m"
#define LOG_INFO_COLOR "\033[1;32m"
#define LOG_DEBUG_COLOR "\033[1;34m"
#define LOG_RESET_COLOR "\033[0m"
#define LOG_BOLD_COLOR "\033[1m"
#define CURRENT_TIME std::chrono::steady_clock::now()

#define LOG_FILE std::cout
#define LOG(x) LOG_FILE << x << std::endl;
#define LOG_INFO(x) LOG_FILE << #x << " = \n" << x << std::endl;
#define LOG_ERROR(x) LOG_FILE << LOG_ERROR_COLOR << x << LOG_RESET_COLOR << std::endl;
#define LOG_WARNING(x) LOG_FILE << LOG_WARNING_COLOR << x << LOG_RESET_COLOR << std::endl;
#define LOG_DEBUG(x) LOG_FILE << LOG_DEBUG_COLOR << x << LOG_RESET_COLOR << std::endl;
#define LOG_LINE(x) \
    LOG_FILE << LOG_BOLD_COLOR << "---------------" << x << "---------------" << LOG_RESET_COLOR << std::endl;
#define LOG_INLINE(x) LOG_FILE << x;

#define START_TIME(x)                                        \
    auto __time_variable = std::chrono::steady_clock::now(); \
    bool __time_log = x;
#define LOG_TIME(x)                                                                                              \
    if (__time_log)                                                                                              \
    {                                                                                                            \
        LOG_FILE << LOG_BOLD_COLOR << x << " time: "                                                             \
                 << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - \
                                                                              __time_variable)                   \
                            .count() *                                                                           \
                        1000                                                                                     \
                 << " ms" << LOG_RESET_COLOR << std::endl;                                                       \
        __time_variable = std::chrono::steady_clock::now();                                                      \
    }

#define TICK_PRECISION 3
#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TICK_INLINE(x) auto bench_inline_##x = std::chrono::steady_clock::now();
#define TOCK(x)                                                                                              \
    std::streamsize ss_##x = std::cout.precision();                                                          \
    std::cout.precision(TICK_PRECISION);                                                                     \
    LOG_FILE << #x ": "                                                                                      \
             << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - \
                                                                          bench_##x)                         \
                    .count()                                                                                 \
             << "s" << std::endl;                                                                            \
    std::cout.precision(ss_##x);
#define TOCK_INLINE(x)                                                                                       \
    std::streamsize ss_##x = std::cout.precision();                                                          \
    std::cout.precision(TICK_PRECISION);                                                                     \
    LOG_FILE << #x ": "                                                                                      \
             << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - \
                                                                          bench_inline_##x)                  \
                    .count()                                                                                 \
             << "s"                                                                                          \
             << "  \t";                                                                                      \
    std::cout.precision(ss_##x);

#define TOCK_VALUE(x) std::chrono::duration<float>(std::chrono::steady_clock::now() - bench_##x).count()

#define APPEND_TIME(T, FUNC, TAG)                        \
    auto bench_##TAG = std::chrono::steady_clock::now(); \
    FUNC;                                                \
    T += std::chrono::duration<float>(std::chrono::steady_clock::now() - bench_##TAG).count();

#define PI 3.14159265359
#define F2I(x) (int)(x + 0.5f)
#define AIR_WAVE_SPEED 340.3f
#define AIR_DENSITY 1.225f
#define CGPU_FUNC __host__ __device__
#define GPU_FUNC __device__
#define CPU_FUNC __host__
#define RAND_F (float)rand() / (float)RAND_MAX
#define RAND_I(min, max) (min + rand() % (max - min + 1))
#define RAND_SIGN (rand() % 2 == 0 ? 1 : -1)
#define BDF2(x) 0.5 * ((x) * (x)-4 * (x) + 3)  // 这是啥？
#define MAX_FLOAT 3.402823466e+38F
#define CHECK_DIR(dir)                          \
    if (!std::filesystem::exists(dir))          \
    {                                           \
        std::filesystem::create_directory(dir); \
    }

typedef float real_t;
const real_t EPS = 1e-8f;
typedef thrust::complex<real_t> cpx;

enum cpx_phase
{
    CPX_REAL,
    CPX_IMAG,
    CPX_ABS,
};

enum PlaneType
{
    XY,
    XZ,
    YZ,
};

typedef std::vector<real_t> RealVec;
typedef std::vector<cpx> ComplexVec;

static inline void checkCudaError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        // printf( "CUDA error: %d : %s at %s:%d \n", err, cudaGetErrorString(err), __FILE__, __LINE__);
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

#define atomicAddCpx(dst, value)                         \
    {                                                    \
        atomicAdd((float *)(dst), (value).real());       \
        atomicAdd(((float *)(dst)) + 1, (value).imag()); \
    }
#define atomicAddCpxBlock(dst, value)                          \
    {                                                          \
        atomicAdd_block((float *)(dst), (value).real());       \
        atomicAdd_block(((float *)(dst)) + 1, (value).imag()); \
    }

#ifdef NDEBUG
#    define cuSafeCall(X) X
#else
#    define cuSafeCall(X) \
        X;                \
        checkCudaError(#X);
#endif

/**
 * @brief Macro to check cuda errors
 *
 */
#ifdef NDEBUG
#    define cuSynchronize() \
        {}
#else
#    define cuSynchronize()                                                                                        \
        {                                                                                                          \
            char str[200];                                                                                         \
            cudaDeviceSynchronize();                                                                               \
            cudaError_t err = cudaGetLastError();                                                                  \
            if (err != cudaSuccess)                                                                                \
            {                                                                                                      \
                sprintf(str, "CUDA error: %d : %s at %s:%d \n", err, cudaGetErrorString(err), __FILE__, __LINE__); \
                throw std::runtime_error(std::string(str));                                                        \
            }                                                                                                      \
        }
#endif

static uint iDivUp(uint a, uint b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
static uint cudaGridSize(uint totalSize, uint blockSize)
{
    int dim = iDivUp(totalSize, blockSize);
    return dim == 0 ? 1 : dim;
}

#define CUDA_BLOCK_SIZE 256
/**
 * @brief Macro definition for execuation of cuda kernels, note that at lease one block will be executed.
 *
 * size: indicate how many threads are required in total.
 * Func: kernel function
 */

#define cuExecute(size, Func, ...)                              \
    {                                                           \
        uint pDims = cudaGridSize((uint)size, CUDA_BLOCK_SIZE); \
        Func<<<pDims, CUDA_BLOCK_SIZE>>>(__VA_ARGS__);          \
        cuSynchronize();                                        \
    }

#define CUDA_3D_BLOCK_SIZE 4
#define cuExecute3D(size, Func, ...)                                                                           \
    {                                                                                                          \
        uint pDimx = cudaGridSize((uint)size.x, CUDA_3D_BLOCK_SIZE);                                           \
        uint pDimy = cudaGridSize((uint)size.y, CUDA_3D_BLOCK_SIZE);                                           \
        uint pDimz = cudaGridSize((uint)size.z, CUDA_3D_BLOCK_SIZE);                                           \
        Func<<<dim3(pDimx, pDimy, pDimz), dim3(CUDA_3D_BLOCK_SIZE, CUDA_3D_BLOCK_SIZE, CUDA_3D_BLOCK_SIZE)>>>( \
            __VA_ARGS__);                                                                                      \
        cuSynchronize();                                                                                       \
    }

#define CUDA_2D_BLOCK_SIZE 8
#define cuExecute2D(size, Func, ...)                                                                   \
    {                                                                                                  \
        uint pDimx = cudaGridSize((uint)size.x, CUDA_2D_BLOCK_SIZE);                                   \
        uint pDimy = cudaGridSize((uint)size.y, CUDA_2D_BLOCK_SIZE);                                   \
        Func<<<dim3(pDimx, pDimy, 1), dim3(CUDA_2D_BLOCK_SIZE, CUDA_2D_BLOCK_SIZE, 1)>>>(__VA_ARGS__); \
        cuSynchronize();                                                                               \
    }

#define cuExecuteBlock(size1, size2, Func, ...) \
    {                                           \
        Func<<<size1, size2>>>(__VA_ARGS__);    \
        cuSynchronize();                        \
    }

#define cuExecuteDyArr(size1, size2, size3, Func, ...) \
    {                                                  \
        Func<<<size1, size2, size3>>>(__VA_ARGS__);    \
        cuSynchronize();                               \
    }

#define SHOW(x) std::cout << #x << ":\n" << x << std::endl;

static std::ostream &operator<<(std::ostream &o, float3 const &f)
{
    return o << "(" << f.x << ", " << f.y << ", " << f.z << ")";
}
static std::ostream &operator<<(std::ostream &o, uint3 const &f)
{
    return o << "(" << f.x << ", " << f.y << ", " << f.z << ")";
}
static std::ostream &operator<<(std::ostream &o, int3 const &f)
{
    return o << "(" << f.x << ", " << f.y << ", " << f.z << ")";
}
static std::ostream &operator<<(std::ostream &o, int4 const &a)
{
    return o << "(" << a.x << ", " << a.y << ", " << a.z << ", " << a.w << ")";
}

class Range
{
    public:
        uint start;
        uint end;
        CGPU_FUNC Range(uint start, uint end) : start(start), end(end) {}
        CGPU_FUNC Range() : start(0), end(0) {}
        inline CGPU_FUNC int length() { return end - start; }
        friend std::ostream &operator<<(std::ostream &os, const Range &r)
        {
            os << "[" << r.start << "," << r.end << ")";
            return os;
        }
        friend bool operator==(const Range &a, const Range &b) { return a.start == b.start && a.end == b.end; }
};

class dim2
{
    public:
        int x, y;
        dim2(int x_, int y_)
        {
            x = x_;
            y = y_;
        }
};

struct is_not_empty
{
        CGPU_FUNC int operator()(const int &x) const { return x != 0; }
};

class BBox
{
    public:
        float3 min;
        float3 max;
        float width;
        BBox() {}
        BBox(float3 _min, float3 _max)
        {
            min = _min;
            max = _max;
            width = std::max(std::max(max.x - min.x, max.y - min.y), max.z - min.z);
        }
        void load_from_txt(std::string filename)
        {
            std::ifstream file(filename);
            file >> min.x >> min.y >> min.z >> max.x >> max.y >> max.z;
            file.close();
            width = std::max(std::max(max.x - min.x, max.y - min.y), max.z - min.z);
        }
        float3 center()
        {
            float3 c;
            c.x = (min.x + max.x) / 2;
            c.y = (min.y + max.y) / 2;
            c.z = (min.z + max.z) / 2;
            return c;
        }
        float length() { return std::max(std::max(max.x - min.x, max.y - min.y), max.z - min.z); }
        friend std::ostream &operator<<(std::ostream &os, const BBox &b)
        {
            os << "[" << b.min << "," << b.max << "]";
            return os;
        }
};

}  // namespace GNNBEM