import sys

sys.path.append("./")

import torch
from src.cuda_imp import (
    ImportanceSampler,
    MonteCarloWeight,
    fast_sparse_matrix_vector_mul,
    fast_sparse_matrix_vector_mul2,
)
from src.timer import Timer
from src.loader.model import ModalSoundObject


obj_id = 2
sound_object = ModalSoundObject(f"dataset/0000{obj_id}")
vertices = torch.tensor(sound_object.vertices, dtype=torch.float32).cuda()
triangles = torch.tensor(sound_object.triangles, dtype=torch.int32).cuda()
importance = torch.ones(len(triangles), dtype=torch.float32).cuda()
sampler = ImportanceSampler(vertices, triangles, importance, 100000)
sampler.update()
sampler.poisson_disk_resample(0.008, 4)
print("sample points: ", sampler.num_samples)
resample_num = 256
G0_constructor = MonteCarloWeight(sampler.points, sampler)
G0_constructor.init_random_states(resample_num)

batch_size = 32
ks = torch.arange(batch_size, dtype=torch.float32) * 0.1 + 10


def run(warm_up=False):
    xs = []
    for k in ks:
        x = torch.randn(sampler.num_samples, dtype=torch.complex64).cuda()
        xs.append(x)

    x_batch = torch.stack(xs).reshape(-1)
    x_batch_fast = torch.stack(xs, dim=1).reshape(-1, batch_size)

    timer = Timer(warm_up == False)
    Gs = []
    ys = []
    for k in ks:
        Gs.append(G0_constructor.get_weights_sparse(resample_num, k))
    timer.log("serialize G")
    for G, x in zip(Gs, xs):
        y = G @ x
        ys.append(y)
    timer.log("serialize mul", record=True)

    G_batch = G0_constructor.get_weights_sparse_ks(resample_num, ks)
    timer.log("batch G", record=True)

    y_batch = G_batch @ x_batch
    timer.log("batch mul", record=True)

    col_indices, values = G0_constructor.get_weights_sparse_ks_fast(resample_num, ks)
    timer.log("batch G fast", record=True)

    y_batch_fast = fast_sparse_matrix_vector_mul(col_indices, values, x_batch_fast)
    timer.log("batch mul fast", record=True)

    y_batch_fast2 = fast_sparse_matrix_vector_mul2(col_indices, values, x_batch_fast)
    timer.log("batch mul fast2", record=True)

    y_batch_fast = y_batch_fast.reshape(-1, batch_size).T
    y_batch_fast2 = y_batch_fast2.reshape(-1, batch_size).T
    y_batch = y_batch.reshape(batch_size, -1)

    for y, y_b, y_b_f, y_b_f2 in zip(ys, y_batch, y_batch_fast, y_batch_fast2):
        assert torch.allclose(y, y_b, rtol=1e-3, atol=1e-5)
        assert torch.allclose(y, y_b_f, rtol=1e-3, atol=1e-5)
        assert torch.allclose(y, y_b_f2, rtol=1e-3, atol=1e-5)


run(True)
run()
