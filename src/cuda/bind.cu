#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include "common.h"
#include "gpu_memory.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "sample.h"
#include "fdtd_warp.h"
#include "sample.h"

using namespace nwob;
#define EPS 1e-3
template <bool deriv>
HOST_DEVICE inline complex Green_func(float3 y, float3 x, float3 xn, float k);

template <>
HOST_DEVICE inline complex Green_func<false>(float3 y, float3 x, float3 xn, float k)
{
    float r = length(x - y);
    if (r < EPS)
        return exp(complex(0, k * r)) / (4 * M_PI);
    return exp(complex(0, k * r)) / (4 * M_PI * r);
}
template <>
HOST_DEVICE inline complex Green_func<true>(float3 y, float3 x, float3 xn, float k)
{
    float r = length(x - y);
    if (r < EPS)
        return 0;
    complex ikr = complex(0, 1) * r * k;
    complex potential = -exp(ikr) / (4 * M_PI * r * r * r) * (1 - ikr) * dot(x - y, xn);
    return potential;
}

template <bool deriv>
void get_monte_carlo_weight(const torch::Tensor trg_points,
                            const torch::Tensor src_points,
                            const torch::Tensor src_normals,
                            const torch::Tensor src_importance,
                            float k,
                            const float cdf_sum,
                            torch::Tensor out)
{
    int N = out.size(0), M = out.size(1);
    GPUMemory<int> near_point_num(N);
    near_point_num.memset(0);
    parallel_for(N * M, [N, M, k, trgs = (float3 *)trg_points.data_ptr(), srcs = (float3 *)src_points.data_ptr(),
                         near_point_num = near_point_num.device_ptr()] __device__(int i) {
        if (length(trgs[i / M] - srcs[i % M]) < EPS)
            atomicAdd(&near_point_num[i / M], 1);
    });

    parallel_for(N * M, [N, M, k, cdf_sum, trgs = (float3 *)trg_points.data_ptr(),
                         srcs = (float3 *)src_points.data_ptr(), src_normals = (float3 *)src_normals.data_ptr(),
                         out = (float2 *)out.data_ptr(), near_point_num = near_point_num.device_ptr(),
                         src_importance = (float *)src_importance.data_ptr()] __device__(int i) {
        int trg_i = i / M, src_i = i % M;
        float3 trg = trgs[trg_i];
        float3 src = srcs[src_i];
        float3 normal = src_normals[src_i];
        auto weight = Green_func<deriv>(trg, src, normal, k);
        float2 result;
        result.x = weight.real();
        result.y = weight.imag();
        float near_field_cdf_sum = M_PI * EPS * EPS * src_importance[src_i];
        if (length(trg - src) < EPS)
        {
            result *= 2 * (2 * M_PI * EPS) / near_point_num[trg_i];
        }
        else
        {
            result *= 2 * (cdf_sum - near_field_cdf_sum) / src_importance[src_i] / (M - near_point_num[trg_i]);
        }
        out[i] = result;
    });
}

template <bool deriv>
torch::Tensor get_monte_carlo_weight_potential_ks(const torch::Tensor trg_points,
                                                  const torch::Tensor src_points,
                                                  const torch::Tensor src_normals,
                                                  const torch::Tensor src_importance,
                                                  const torch::Tensor ks,
                                                  const float cdf_sum)
{
    int N = trg_points.size(0), M = src_points.size(0);
    int batch_size = ks.size(0);
    torch::Tensor out = torch::empty({batch_size, N, M}, torch::dtype(torch::kComplexFloat).device(torch::kCUDA));
    parallel_for(N * M,
                 [N, M, batch_size, cdf_sum, trgs = (float3 *)trg_points.data_ptr(),
                  srcs = (float3 *)src_points.data_ptr(), src_normals = (float3 *)src_normals.data_ptr(),
                  out = PitchedPtr<float2, 3>((float2 *)out.data_ptr(), batch_size, N, M),
                  src_importance = (float *)src_importance.data_ptr(), ks = (float *)ks.data_ptr()] __device__(int i) {
                     int trg_i = i / M, src_i = i % M;
                     float3 trg = trgs[trg_i];
                     float3 src = srcs[src_i];
                     float3 normal = src_normals[src_i];
                     float W = cdf_sum / src_importance[src_i] / M;
                     for (int k_i = 0; k_i < batch_size; k_i++)
                     {
                         auto weight = Green_func<deriv>(trg, src, normal, ks[k_i]);
                         out(k_i, trg_i, src_i).x = weight.real() * W;
                         out(k_i, trg_i, src_i).y = weight.imag() * W;
                     }
                 });
    return out;
}

template <bool deriv>
void get_monte_carlo_weight_boundary(const torch::Tensor trg_points,
                                     const torch::Tensor src_points,
                                     const torch::Tensor src_normals,
                                     const torch::Tensor src_importance,
                                     float k,
                                     const float cdf_sum,
                                     torch::Tensor out)
{
    int N = out.size(0), M = out.size(1);
    parallel_for(
        N * M, [N, M, k, cdf_sum, trgs = (float3 *)trg_points.data_ptr(), srcs = (float3 *)src_points.data_ptr(),
                src_normals = (float3 *)src_normals.data_ptr(), out = (float2 *)out.data_ptr(),
                src_importance = (float *)src_importance.data_ptr()] __device__(int i) {
            int trg_i = i / M, src_i = i % M;
            float3 trg = trgs[trg_i];
            float3 src = srcs[src_i];
            float3 normal = src_normals[src_i];
            auto weight = Green_func<deriv>(trg, src, normal, k);
            float2 result;
            result.x = weight.real();
            result.y = weight.imag();
            if (trg_i == src_i)
            {
                result *= 2 * (2 * M_PI * EPS);
            }
            else
            {
                result *= 2 * (cdf_sum - M_PI * EPS * EPS * src_importance[src_i]) / src_importance[src_i] / (M - 1);
            }
            out[i] = result;
        });
}

template <bool deriv>
torch::Tensor get_monte_carlo_weight_boundary_ks(const torch::Tensor trg_points,
                                                 const torch::Tensor src_points,
                                                 const torch::Tensor src_normals,
                                                 const torch::Tensor src_importance,
                                                 const torch::Tensor ks,
                                                 const float cdf_sum)
{
    int N = trg_points.size(0), M = src_points.size(0);
    int batch_size = ks.size(0);
    torch::Tensor out = torch::empty({batch_size, N, M}, torch::dtype(torch::kComplexFloat).device(torch::kCUDA));
    parallel_for(N * M,
                 [N, M, batch_size, cdf_sum, trgs = (float3 *)trg_points.data_ptr(),
                  srcs = (float3 *)src_points.data_ptr(), src_normals = (float3 *)src_normals.data_ptr(),
                  out = PitchedPtr<float2, 3>((float2 *)out.data_ptr(), batch_size, N, M),
                  src_importance = (float *)src_importance.data_ptr(), ks = (float *)ks.data_ptr()] __device__(int i) {
                     int trg_i = i / M, src_i = i % M;
                     float3 trg = trgs[trg_i];
                     float3 src = srcs[src_i];
                     float3 normal = src_normals[src_i];
                     float W;
                     if (trg_i == src_i)
                     {
                         W = 2 * (2 * M_PI * EPS);
                     }
                     else
                     {
                         W = 2 * (cdf_sum - M_PI * EPS * EPS * src_importance[src_i]) / src_importance[src_i] / (M - 1);
                     }
                     for (int k_i = 0; k_i < batch_size; k_i++)
                     {
                         auto weight = Green_func<deriv>(trg, src, normal, ks[k_i]);
                         out(k_i, trg_i, src_i).x = weight.real() * W;
                         out(k_i, trg_i, src_i).y = weight.imag() * W;
                     }
                 });
    return out;
}

class Reservoir
{
    public:
        int y_idx;
        float w_sum;
        int M;
        HOST_DEVICE inline Reservoir()
        {
            y_idx = -1;
            w_sum = 0;
            M = 0;
        }

        HOST_DEVICE inline void update(int x_idx, float w_i, float rnd)
        {
            w_sum += w_i;
            M++;
            if (rnd <= w_i / w_sum)
            {
                y_idx = x_idx;
            }
        }
};

template <bool deriv>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_monte_carlo_weight_sparse(const torch::Tensor points_,
                                                                                      const torch::Tensor normals_,
                                                                                      const torch::Tensor importance_,
                                                                                      const torch::Tensor states_,
                                                                                      float k,
                                                                                      const float cdf_sum,
                                                                                      int resample_num)
{
    int N = points_.size(0);
    auto row_indices_ = torch::empty({N + 1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto col_indices_ = torch::empty({N * resample_num}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto values_ = torch::empty({N * resample_num, 2}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto row_indices = (int *)row_indices_.data_ptr();
    auto col_indices = PitchedPtr<int, 2>((int *)col_indices_.data_ptr(), N, resample_num);
    auto values = PitchedPtr<float2, 2>((float2 *)values_.data_ptr(), N, resample_num);
    auto points = (float3 *)points_.data_ptr();
    auto normals = (float3 *)normals_.data_ptr();
    auto importance = (float *)importance_.data_ptr();
    int stride = ceil(N / resample_num);
    auto states = PitchedPtr<randomState, 2>((randomState *)states_.data_ptr(), N, resample_num);
    parallel_for(N * resample_num, [=] __device__(int i) {
        int row_i = i / resample_num, col_i = i % resample_num;
        int trg_i = row_i, start_idx = col_i * stride;
        int end_idx = min(start_idx + stride, N);
        if (col_i > N - (stride - 1) * resample_num)
        {
            start_idx += col_i - (N - (stride - 1) * resample_num);
            end_idx = min(start_idx + stride - 1, N);
        }
        float3 trg = points[trg_i];
        Reservoir reservoir;
        auto state = states(trg_i, col_i);
        float p = 1.0f / (end_idx - start_idx);
        for (int src_i = start_idx; src_i < end_idx; src_i++)
        {
            float p_hat = 1.0f / (length(trg - points[src_i]) + EPS);
            reservoir.update(src_i, p_hat / p, curand_uniform(&state));
        }
        auto src_i = reservoir.y_idx;
        auto green_func_value = Green_func<deriv>(trg, points[src_i], normals[src_i], k);
        float2 W = make_float2(green_func_value.real(), green_func_value.imag());
        W *= (length(trg - points[src_i]) + EPS) * (1.0f / reservoir.M * reservoir.w_sum);
        if (trg_i == src_i)
        {
            W *= 2 * (2 * M_PI * EPS);
        }
        else
        {
            W *= 2 * (cdf_sum - M_PI * EPS * EPS * importance[src_i]) / importance[src_i] / (N - 1);
        }
        values(trg_i, col_i) = W;
        col_indices(trg_i, col_i) = reservoir.y_idx;
        if (i < N + 1)
            row_indices[i] = i * resample_num;
    });
    return std::make_tuple(row_indices_, col_indices_, values_);
}

template <bool deriv>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_monte_carlo_weight_sparse_ks(
    const torch::Tensor points_,
    const torch::Tensor normals_,
    const torch::Tensor importance_,
    const torch::Tensor states_,
    const torch::Tensor ks_,
    const float cdf_sum,
    int resample_num)
{
    int N = points_.size(0);
    int M = ks_.size(0);
    auto row_indices_ = torch::empty({N + 1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto col_indices_ = torch::empty({N * resample_num}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto values_ = torch::empty({N * resample_num, M, 2}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto row_indices = (int *)row_indices_.data_ptr();
    auto col_indices = PitchedPtr<int, 2>((int *)col_indices_.data_ptr(), N, resample_num);
    auto values = PitchedPtr<float2, 3>((float2 *)values_.data_ptr(), N, resample_num, M);
    auto points = (float3 *)points_.data_ptr();
    auto normals = (float3 *)normals_.data_ptr();
    auto importance = (float *)importance_.data_ptr();
    auto ks = (float *)ks_.data_ptr();
    int stride = ceil(N / resample_num);
    auto states = PitchedPtr<randomState, 2>((randomState *)states_.data_ptr(), N, resample_num);
    parallel_for(N * resample_num, [=] __device__(int i) {
        int row_i = i / resample_num, col_i = i % resample_num;
        int trg_i = row_i, start_idx = col_i * stride;
        int end_idx = min(start_idx + stride, N);
        if (col_i > N - (stride - 1) * resample_num)
        {
            start_idx += col_i - (N - (stride - 1) * resample_num);
            end_idx = min(start_idx + stride - 1, N);
        }
        float3 trg = points[trg_i];
        Reservoir reservoir;
        auto state = states(trg_i, col_i);
        float p = 1.0f / (end_idx - start_idx);
        for (int src_i = start_idx; src_i < end_idx; src_i++)
        {
            float p_hat = 1.0f / (length(trg - points[src_i]) + EPS);
            reservoir.update(src_i, p_hat / p, curand_uniform(&state));
        }
        auto src_i = reservoir.y_idx;
        auto src = points[src_i];
        auto normal = normals[src_i];
        float W = (length(trg - points[src_i]) + EPS) * (1.0f / reservoir.M * reservoir.w_sum);
        if (trg_i == src_i)
        {
            W *= 2 * (2 * M_PI * EPS);
        }
        else
        {
            W *= 2 * (cdf_sum - M_PI * EPS * EPS * importance[src_i]) / importance[src_i] / (N - 1);
        }
        for (int k_i = 0; k_i < M; k_i++)
        {
            float k = ks[k_i];
            auto weight = Green_func<deriv>(trg, src, normal, k);
            values(trg_i, col_i, k_i).x = weight.real() * W;
            values(trg_i, col_i, k_i).y = weight.imag() * W;
        }
        col_indices(trg_i, col_i) = reservoir.y_idx;
        if (i < N + 1)
            row_indices[i] = i * resample_num;
    });
    return std::make_tuple(row_indices_, col_indices_, values_);
}

torch::Tensor sparse_matrix_vector_mul(const torch::Tensor col_indices_,
                                       const torch::Tensor values_,
                                       const torch::Tensor x_)
{
    auto N = x_.size(0);
    auto batch_size = x_.size(1);
    auto resample_num = col_indices_.size(0) / N;
    auto y_ = torch::empty({N, batch_size}, torch::dtype(torch::kComplexFloat).device(torch::kCUDA));
    auto col_indices = PitchedPtr<int, 2>((int *)col_indices_.data_ptr(), N, resample_num);
    using cpx = thrust::complex<float>;
    auto values = PitchedPtr<cpx, 3>((cpx *)values_.data_ptr(), N, resample_num, batch_size);
    auto x = PitchedPtr<cpx, 2>((cpx *)x_.data_ptr(), N, batch_size);
    auto y = PitchedPtr<cpx, 2>((cpx *)y_.data_ptr(), N, batch_size);
    parallel_for(N * batch_size, [=] __device__(int i) {
        int row_i = i / batch_size, k_i = i % batch_size;
        cpx sum = cpx(0.f, 0.f);
        for (int j = 0; j < resample_num; j++)
        {
            int col_i = col_indices(row_i, j);
            sum += values(row_i, j, k_i) * x(col_i, k_i);
        }
        y(row_i, k_i) = sum;
    });
    return y_;
}

torch::Tensor sparse_matrix_vector_mul_fast(const torch::Tensor col_indices_,
                                            const torch::Tensor values_,
                                            const torch::Tensor x_)
{
    auto N = x_.size(0);
    auto batch_size = x_.size(1);
    auto resample_num = col_indices_.size(0) / N;
    auto y_ = torch::empty({N, batch_size}, torch::dtype(torch::kComplexFloat).device(torch::kCUDA));
    auto col_indices = PitchedPtr<int, 2>((int *)col_indices_.data_ptr(), N, resample_num);
    using cpx = thrust::complex<float>;
    auto values = PitchedPtr<cpx, 3>((cpx *)values_.data_ptr(), N, resample_num, batch_size);
    auto x = PitchedPtr<cpx, 2>((cpx *)x_.data_ptr(), N, batch_size);
    auto y = PitchedPtr<cpx, 2>((cpx *)y_.data_ptr(), N, batch_size);

    size_t shmem_size = sizeof(cpx) * batch_size;
    parallel_for_block(shmem_size, N, resample_num, [=] __device__(int block_i, int thread_i) {
        extern __shared__ cpx shared_sum[];
        int row_i = block_i;
        if (thread_i < batch_size)
            shared_sum[thread_i] = cpx(0.f, 0.f);
        __syncthreads();
        int col_i = col_indices(row_i, thread_i);
        for (int j = 0; j < batch_size; j++)
        {
            int idx = (j + thread_i) % batch_size;
            cpx item = values(row_i, thread_i, idx) * x(col_i, idx);
            atomicAddCpxBlock(&shared_sum[idx], item);
        }
        __syncthreads();
        if (thread_i < batch_size)
            y(row_i, thread_i) = shared_sum[thread_i];
    });
    return y_;
}

#include "poisson.h"
#include "multipole.h"

template <bool deriv, int N>
torch::Tensor get_multipole_values(torch::Tensor x0_, torch::Tensor n0_, torch::Tensor x_, torch::Tensor n_, float k)
{
    static_assert(std::is_same<std::integral_constant<int, N>, std::integral_constant<int, 0>>::value ||
                      std::is_same<std::integral_constant<int, N>, std::integral_constant<int, 1>>::value,
                  "N must be either 0 or 1.");
    float3 x0 = make_float3(x0_[0].item<float>(), x0_[1].item<float>(), x0_[2].item<float>());
    float3 n0 = make_float3(n0_[0].item<float>(), n0_[1].item<float>(), n0_[2].item<float>());
    auto x = (float3 *)x_.data_ptr();
    auto n = (float3 *)n_.data_ptr();
    auto out_ = torch::empty({x_.size(0)}, torch::dtype(torch::kComplexFloat).device(torch::kCUDA));
    auto out = (complex *)out_.data_ptr();
    parallel_for(x_.size(0), [=] __device__(int i) {
        float3 x_i = x[i];
        float3 n_i = n[i];
        out[i] = multipole<N, deriv>(x0, x_i, n0, n_i, k);
    });
    return out_;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_random_states", &get_random_states, "");
    m.def("get_cdf", &get_cdf, "");
    m.def("importance_sample", &importance_sample, "");
    m.def("poisson_disk_resample", &poisson_disk_resample, "");
    m.def("allocate_grids_data", &allocate_grids_data, "");
    m.def("FDTD_simulation", &FDTD_simulation, "");
    m.def("get_monte_carlo_weight1", &get_monte_carlo_weight<true>, "");
    m.def("get_monte_carlo_weight0", &get_monte_carlo_weight<false>, "");
    m.def("get_monte_carlo_weight_potential_ks1", &get_monte_carlo_weight_potential_ks<true>, "");
    m.def("get_monte_carlo_weight_potential_ks0", &get_monte_carlo_weight_potential_ks<false>, "");
    m.def("get_monte_carlo_weight_boundary1", &get_monte_carlo_weight_boundary<true>, "");
    m.def("get_monte_carlo_weight_boundary0", &get_monte_carlo_weight_boundary<false>, "");
    m.def("get_monte_carlo_weight_boundary_ks1", &get_monte_carlo_weight_boundary_ks<true>, "");
    m.def("get_monte_carlo_weight_boundary_ks0", &get_monte_carlo_weight_boundary_ks<false>, "");
    m.def("get_monte_carlo_weight_sparse1", &get_monte_carlo_weight_sparse<true>, "");
    m.def("get_monte_carlo_weight_sparse0", &get_monte_carlo_weight_sparse<false>, "");
    m.def("get_monte_carlo_weight_sparse_ks1", &get_monte_carlo_weight_sparse_ks<true>, "");
    m.def("get_monte_carlo_weight_sparse_ks0", &get_monte_carlo_weight_sparse_ks<false>, "");
    m.def("sparse_matrix_vector_mul", &sparse_matrix_vector_mul, "");
    m.def("sparse_matrix_vector_mul_fast", &sparse_matrix_vector_mul_fast, "");
    m.def("get_multipole_values_0_deriv", &get_multipole_values<true, 0>, "");
    m.def("get_multipole_values_1_deriv", &get_multipole_values<true, 1>, "");
    m.def("get_multipole_values_0", &get_multipole_values<false, 0>, "");
    m.def("get_multipole_values_1", &get_multipole_values<false, 1>, "");
}