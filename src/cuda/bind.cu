#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include "common.h"
#include "gpu_memory.h"
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <tuple>
#include <vector>
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

class SamplePoint
{
    public:
        float3 p, n;
        float neumann;
        float importance;
        void inline HOST_DEVICE print(int indent = 0)
        {
            char prefix[100];
            for (int i = 0; i < indent; i++)
                prefix[i] = ' ';
            prefix[indent] = '\0';
            printf("%sp: %f %f %f\n", prefix, p.x, p.y, p.z);
            printf("%sn: %f %f %f\n", prefix, n.x, n.y, n.z);
            printf("%sneumann: %f\n", prefix, neumann);
            printf("%simportance: %f\n", prefix, importance);
        }
};

class AliasItem
{
    public:
        float3 v0, v1, v2, n;
        float neumann;
        float probability;
        int neighbor_left;
        int neighbor_right;

        float inline HOST_DEVICE get_area() { return length(cross(v1 - v0, v2 - v0)) / 2; }

        float inline HOST_DEVICE get_point_probability()
        {
#ifdef NAN_DEBUG
            if (IS_NAN_INF(probability / get_area()))
            {
                printf(
                    "probability / get_area() is nan with probability = %f, get_area() = %f, v0 = %f %f %f, v1 = %f %f "
                    "%f, v2 = %f %f %f\n",
                    probability, get_area(), v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
            }
#endif
            return probability / get_area();
        }

        SamplePoint inline __device__ sample(randomState &state)
        {
            float u = curand_uniform(&state);
            float v = curand_uniform(&state);
            if (u + v > 1.f)
            {
                u = 1.f - u;
                v = 1.f - v;
            }
            float w = 1.f - u - v;
            float3 p = v0 * u + v1 * v + v2 * w;
            SamplePoint sp;
            sp.p = p;
            sp.n = n;
            sp.neumann = neumann;
#ifdef NAN_DEBUG
            if (IS_NAN_INF(sp.p.x) || IS_NAN_INF(sp.p.y) || IS_NAN_INF(sp.p.z) || abs(sp.p.x) > 1e2 ||
                abs(sp.p.y) > 1e2 || abs(sp.p.z) > 1e2)
            {
                printf("sp.p is nan with u = %f, v = %f, w = %f, v0 = %f %f %f, v1 = %f %f %f, v2 = %f %f %f\n", u, v,
                       w, v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
            }
#endif
            return sp;
        }

        void HOST_DEVICE inline print(int indent = 0)
        {
            char prefix[100];
            for (int i = 0; i < indent; i++)
                prefix[i] = ' ';
            prefix[indent] = '\0';
            printf("%sv0: %f %f %f\n", prefix, v0.x, v0.y, v0.z);
            printf("%sv1: %f %f %f\n", prefix, v1.x, v1.y, v1.z);
            printf("%sv2: %f %f %f\n", prefix, v2.x, v2.y, v2.z);
            printf("%sn: %f %f %f\n", prefix, n.x, n.y, n.z);
            printf("%sneumann: %f\n", prefix, neumann);
            printf("%sprobability: %f\n", prefix, probability);
            printf("%sneighbor_left: %d\n", prefix, neighbor_left);
            printf("%sneighbor_right: %d\n", prefix, neighbor_right);
        }
};

torch::Tensor get_sample_table(const torch::Tensor vertices,
                               const torch::Tensor triangles,
                               const torch::Tensor triangle_importance,
                               const torch::Tensor triangle_neumann,
                               const int alias_factor)
{
    int vertices_size = vertices.size(0);
    int triangles_size = triangles.size(0);
    torch::Tensor discrete_prob = torch::empty({triangles_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    parallel_for(triangles_size, [vertices = (float3 *)vertices.data_ptr(), triangles = (int3 *)triangles.data_ptr(),
                                  discrete_prob = (float *)discrete_prob.data_ptr(),
                                  triangle_importance = (float *)triangle_importance.data_ptr()] __device__(int i) {
        float3 v0 = vertices[triangles[i].x];
        float3 v1 = vertices[triangles[i].y];
        float3 v2 = vertices[triangles[i].z];
        discrete_prob[i] = length(cross(v1 - v0, v2 - v0)) / 2 * triangle_importance[i];
    });
    auto min_prob = torch::min(discrete_prob).item<float>();
    torch::Tensor discrete_prob_int = torch::empty({triangles_size}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    torch::Tensor discrete_prob_int_cdf =
        torch::empty({triangles_size}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    parallel_for(triangles_size,
                 [discrete_prob = (float *)discrete_prob.data_ptr(),
                  discrete_prob_int = (int *)discrete_prob_int.data_ptr(), min_prob, alias_factor] __device__(int i) {
                     discrete_prob_int[i] = (int)(discrete_prob[i] / min_prob * alias_factor);
                 });
    thrust::inclusive_scan(thrust::device, (int *)discrete_prob_int.data_ptr(),
                           (int *)discrete_prob_int.data_ptr() + triangles_size,
                           (int *)discrete_prob_int_cdf.data_ptr());
    int alias_table_size = discrete_prob_int_cdf[triangles_size - 1].item<int>();
    torch::Tensor alias_table = torch::empty({alias_table_size, sizeof(AliasItem) / sizeof(int)},
                                             torch::dtype(torch::kFloat32).device(torch::kCUDA));
    parallel_for(
        alias_table_size,
        [alias_table = (AliasItem *)alias_table.data_ptr(), discrete_prob_int = (int *)discrete_prob_int.data_ptr(),
         discrete_prob_int_cdf = (int *)discrete_prob_int_cdf.data_ptr(), vertices = (float3 *)vertices.data_ptr(),
         triangles = (int3 *)triangles.data_ptr(), triangle_neumann = (float *)triangle_neumann.data_ptr(),
         triangles_size] __device__(int i) {
            int x = i + 1;
            // binary search
            uint l = 0, r = triangles_size - 1;
            while (l < r)
            {
                uint mid = (l + r) / 2;
                if (discrete_prob_int_cdf[mid] < x)
                    l = mid + 1;
                else
                    r = mid;
            }
            uint element_id = l;
            float3 v0 = vertices[triangles[element_id].x];
            float3 v1 = vertices[triangles[element_id].y];
            float3 v2 = vertices[triangles[element_id].z];
            alias_table[i].v0 = v0;
            alias_table[i].v1 = v1;
            alias_table[i].v2 = v2;
            alias_table[i].n = normalize(cross(v1 - v0, v2 - v0));
            alias_table[i].neumann = triangle_neumann[element_id];
            alias_table[i].probability =
                (float)discrete_prob_int[element_id] / (float)discrete_prob_int_cdf[triangles_size - 1];
            alias_table[i].neighbor_left = element_id == 0 ? 0 : discrete_prob_int_cdf[element_id - 1];
            alias_table[i].neighbor_right = discrete_prob_int_cdf[element_id];
        });
    return alias_table;
}

void print_alias_table(const torch::Tensor alias_table)
{
    parallel_for(1, [alias_table = (AliasItem *)alias_table.data_ptr(),
                     alias_table_size = alias_table.size(0)] __device__(int i) {
        for (int j = 0; j < alias_table_size; j++)
        {
            printf("alias_table[%d]:\n", j);
            alias_table[j].print(2);
        }
    });
}

class Reservoir
{
    public:
        SamplePoint y;
        float w_sum;
        int M;
        float W;
        void inline __device__ update(SamplePoint x, float w_i, randomState &state)
        {
            w_sum += w_i;
            M++;
            if (curand_uniform(&state) * w_sum <= w_i)
            {
                y = x;
            }
        }

        void inline HOST_DEVICE reset()
        {
            w_sum = 0;
            M = 0;
        }

        void inline HOST_DEVICE print(int indent = 0)
        {
            char prefix[100];
            for (int i = 0; i < indent; i++)
                prefix[i] = ' ';
            prefix[indent] = '\0';
            printf("%sw_sum: %f\n", prefix, w_sum);
            printf("%sM: %d\n", prefix, M);
            printf("%sW: %f\n", prefix, W);
            printf("%sy:\n", prefix);
            y.print(indent + 2);
        }
};

class RamplePointPair
{
    public:
        SamplePoint trg;
        Reservoir src;
        Reservoir reused_src;
        randomState state;

        void inline HOST_DEVICE print(int indent = 0)
        {
            char prefix[100];
            for (int i = 0; i < indent; i++)
                prefix[i] = ' ';
            prefix[indent] = '\0';
            printf("%strg:\n", prefix);
            trg.print(indent + 2);
            printf("%sstate: %d\n", prefix, state);
            printf("%ssrc:\n", prefix);
            src.print(indent + 2);
            printf("%sreused_src:\n", prefix);
            reused_src.print(indent + 2);
        }
};

torch::Tensor allocate_sample_memory(const torch::Tensor alias_table)
{
    auto num_samples = alias_table.size(0);
    torch::Tensor point_pairs = torch::empty({num_samples, sizeof(RamplePointPair) / sizeof(float)},
                                             torch::dtype(torch::kFloat32).device(torch::kCUDA));
    // initialize points
    GPUMemory<unsigned long long> seeds(num_samples);
    seeds.copy_from_host(get_random_seeds(num_samples));
    parallel_for(num_samples, [point_pairs = (RamplePointPair *)point_pairs.data_ptr(),
                               seeds = seeds.device_ptr()] __device__(int i) {
        curand_init(seeds[i], 0, 0, &point_pairs[i].state);
        point_pairs[i].src.reset();
        point_pairs[i].reused_src.reset();
    });
    return point_pairs;
}

void sample_alias_table(const torch::Tensor alias_table, torch::Tensor point_pairs, int reuse_num, int candidate_num)
{
    auto num_samples = point_pairs.size(0);
    parallel_for(num_samples, [reuse_num, alias_table = (AliasItem *)alias_table.data_ptr(),
                               point_pairs = (RamplePointPair *)point_pairs.data_ptr()] __device__(int i) {
        AliasItem &item = alias_table[i];
        // Sample new target points
        RamplePointPair &point_pair = point_pairs[i];
        auto trg = item.sample(point_pair.state);
        point_pair.trg = trg;
#ifdef NAN_DEBUG
        if (length(trg.p) > 10 || length(trg.n) > 10)
        {
            printf("trg.p is nan with trg.p = %f %f %f, trg.n = %f %f %f\n", trg.p.x, trg.p.y, trg.p.z, trg.n.x,
                   trg.n.y, trg.n.z);
        }
#endif
        // Combining the streams of neighbor reservoirs for reuse
        Reservoir s;
        s.reset();
        int neighbor_size = item.neighbor_right - item.neighbor_left;
        int start = (i - item.neighbor_left);
        int s_M = 0;
        for (int j = 0; j < reuse_num; j++)
        {
            int index = (start + j) % neighbor_size + item.neighbor_left;
            Reservoir &r = point_pairs[index].src;
            float distance = max(length(trg.p - r.y.p), EPS);
            float p_hat_r = 1.f / distance;
            s.update(r.y, p_hat_r * r.W * r.M, point_pair.state);
            s_M += r.M;
        }
        s.M = s_M;
        float p_hat_s = 1.f / max(length(trg.p - s.y.p), EPS);
        s.W = 1 / (p_hat_s * s.M) * s.w_sum;
        point_pair.reused_src = s;

#ifdef NAN_DEBUG
        if (length(point_pair.trg.p) > 10 || length(point_pair.trg.n) > 10)
        {
            printf("point_pair.trg.p.y is nan with point_pair.trg.p = %f %f %f, point_pair.trg.n = %f %f %f\n",
                   point_pair.trg.p.x, point_pair.trg.p.y, point_pair.trg.p.z, point_pair.trg.n.x, point_pair.trg.n.y,
                   point_pair.trg.n.z);
        }
#endif
    });

    // Sample new src points and update the reservoir
    parallel_for(num_samples, [candidate_num, alias_table_size = num_samples,
                               alias_table = (AliasItem *)alias_table.data_ptr(),
                               point_pairs = (RamplePointPair *)point_pairs.data_ptr()] __device__(int i) {
        RamplePointPair &point_pair = point_pairs[i];
        Reservoir r = point_pair.reused_src;
        for (int j = 0; j < candidate_num; j++)
        {
            int alias_index =
                min((int)(curand_uniform(&point_pair.state) * alias_table_size), (int)alias_table_size - 1);
            AliasItem &item = alias_table[alias_index];
            SamplePoint x = item.sample(point_pair.state);
            float distance = max(length(x.p - point_pair.trg.p), EPS);
            float p_hat_x = 1.f / distance;
            float p_x = item.get_point_probability();
#ifdef NAN_DEBUG
            if (IS_NAN_INF(p_hat_x / p_x))
            {
                printf("alias_index: %d, p_hat_x / p_x is nan with p_hat_x = %f, p_x = %f\n", alias_index, p_hat_x,
                       p_x);
            }
#endif
            r.update(x, p_hat_x / p_x, point_pair.state);
        }
        float p_hat_r = 1.f / max(length(r.y.p - point_pair.trg.p), EPS);
        r.W = 1 / (p_hat_r * r.M) * r.w_sum;
#ifdef NAN_DEBUG
        if (IS_NAN_INF(r.W))
        {
            printf(
                "r.W is nan with p_hat_r = %f, r.M = %d, r.w_sum = %f, r.y.p = %f %f %f, point_pair.trg.p = %f %f %f\n",
                p_hat_r, r.M, r.w_sum, r.y.p.x, r.y.p.y, r.y.p.z, point_pair.trg.p.x, point_pair.trg.p.y,
                point_pair.trg.p.z);
        }
#endif
        point_pair.src = r;
    });
}

void print_point_pairs(const torch::Tensor point_pairs)
{
    parallel_for(1, [point_pairs = (RamplePointPair *)point_pairs.data_ptr(),
                     point_pairs_size = point_pairs.size(0)] __device__(int i) {
        for (int j = 0; j < point_pairs_size; j++)
        {
            printf("point_pairs[%d]:\n", j);
            point_pairs[j].print(2);
        }
    });
}

// dirichlet_trg = A * dirichlet_src + B
void get_equation_AB(const torch::Tensor point_pairs,
                     const torch::Tensor src_points,
                     const torch::Tensor trg_points,
                     const torch::Tensor A,
                     const torch::Tensor B,
                     float wave_number,
                     const torch::Tensor bbox_min,
                     float bbox_max_len)
{
    auto num_samples = point_pairs.size(0);
    float3 bbox_min_pos = {bbox_min[0].item<float>(), bbox_min[1].item<float>(), bbox_min[2].item<float>()};
    parallel_for(
        num_samples, [wave_number, bbox_min_pos, bbox_max_len, point_pairs = (RamplePointPair *)point_pairs.data_ptr(),
                      src_points = (float3 *)src_points.data_ptr(), trg_points = (float3 *)trg_points.data_ptr(),
                      A = (float *)A.data_ptr(), B = (float *)B.data_ptr()] __device__(int i) {
            auto &p = point_pairs[i];
            src_points[i * 2] = (p.src.y.p + bbox_min_pos) / bbox_max_len;
            src_points[i * 2 + 1] = p.src.y.n;
            trg_points[i * 2] = (p.trg.p + bbox_min_pos) / bbox_max_len;
            trg_points[i * 2 + 1] = p.trg.n;
            A[i] = p.src.W * Green_func<true>(p.trg.p, p.src.y.p, p.src.y.n, wave_number).real();
#ifdef NAN_DEBUG
            if (IS_NAN_INF(A[i]))
            {
                printf(
                    "A[%d] is nan with W = %f, Green_func = %f, p.trg.p = %f %f %f, p.src.y.p = %f %f %f, p.src.y.n = "
                    "%f %f %f\n",
                    i, p.src.W, Green_func<true>(p.trg.p, p.src.y.p, p.src.y.n, wave_number).real(), p.trg.p.x,
                    p.trg.p.y, p.trg.p.z, p.src.y.p.x, p.src.y.p.y, p.src.y.p.z, p.src.y.n.x, p.src.y.n.y, p.src.y.n.z);
            }
#endif
            B[i] = -p.src.W * Green_func<false>(p.trg.p, p.src.y.p, p.src.y.n, wave_number).real() * p.src.y.neumann;
#ifdef NAN_DEBUG
            if (IS_NAN_INF(B[i]))
            {
                printf(
                    "B[%d] is nan with W = %f, Green_func = %f, p.trg.p = %f %f %f, p.src.y.p = %f %f %f, p.src.y.n = "
                    "%f %f %f, p.src.y.neumann = %f\n",
                    i, p.src.W, Green_func<false>(p.trg.p, p.src.y.p, p.src.y.n, wave_number).real(), p.trg.p.x,
                    p.trg.p.y, p.trg.p.z, p.src.y.p.x, p.src.y.p.y, p.src.y.p.z, p.src.y.n.x, p.src.y.n.y, p.src.y.n.z,
                    p.src.y.neumann);
            }
#endif
        });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_sample_table", &get_sample_table, "");
    m.def("print_alias_table", &print_alias_table, "");
    m.def("allocate_sample_memory", &allocate_sample_memory, "");
    m.def("sample_alias_table", &sample_alias_table, "");
    m.def("print_point_pairs", &print_point_pairs, "");
    m.def("get_equation_AB", &get_equation_AB, "");
}