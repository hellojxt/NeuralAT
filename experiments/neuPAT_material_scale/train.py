import sys

sys.path.append("./")
from src.net.model import NeuPAT
import torch
from glob import glob
import torch
from tqdm import tqdm
import json
import os

data_dir = "dataset/NeuPAT_new/scale/baseline"
with open(f"{data_dir}/net.json", "r") as file:
    net_config = json.load(file)

with open(f"{data_dir}/../config.json", "r") as file:
    js = json.load(file)
    sample_config = js.get("sample", {})
    obj_config = js.get("vibration_obj", {})
    size_base = obj_config.get("size")

data = torch.load(f"{data_dir}/../modal_data.pt")
vertices_base = data["vertices"]
triangles = data["triangles"]
neumann_vtx = data["neumann_vtx"]
ks_base = data["ks"]
mode_num = len(ks_base)
mode_num = 60

ks_base = ks_base[:mode_num]
neumann_vtx = neumann_vtx[:mode_num]

freq_rate = sample_config.get("freq_rate")
size_rate = sample_config.get("size_rate")
bbox_rate = sample_config.get("bbox_rate")
sample_num = sample_config.get("sample_num")
point_num_per_sample = sample_config.get("point_num_per_sample")


data_points_lst = glob(f"{data_dir}/../data/*.pt")
xs = []
ys = []
mode_num = 60
for data_points in data_points_lst:
    data = torch.load(data_points)
    x = data["x"].reshape(-1, 5)
    x = torch.cat([x, (x[..., -1] * x[..., -2]).unsqueeze(-1)], dim=-1)
    y = data["y"].permute(0, 2, 1)[:, :, :mode_num].reshape(-1, mode_num)
    xs.append(x)
    ys.append(y)

xs = torch.cat(xs, dim=0)
ys = torch.cat(ys, dim=0)
ys = ((ys + 10e-6) / 10e-6).log10()
print("xs", xs.shape)
print("ys", ys.shape)

xs_train = xs[: int(len(xs) * 0.8)].cuda()
ys_train = ys[: int(len(ys) * 0.8)].cuda()
xs_test = xs[int(len(xs) * 0.8) :].cuda()
ys_test = ys[int(len(ys) * 0.8) :].cuda()
del xs, ys


train_params = net_config.get("train", {})
batch_size = train_params.get("batch_size")

from torch.utils.tensorboard import SummaryWriter

logs = glob(f"{data_dir}/events*")
for log in logs:
    os.remove(log)
writer = SummaryWriter(log_dir=data_dir)


model = NeuPAT(mode_num, net_config).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=train_params.get("lr"))
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=train_params.get("step_size"), gamma=train_params.get("gamma")
)

# torch.autograd.set_detect_anomaly(True)
max_epochs = train_params.get("max_epochs")
test_step = train_params.get("test_step")
for epoch_idx in tqdm(range(max_epochs)):
    # Create batches manually
    indices = torch.randperm(xs_train.size(0), device="cuda")

    loss_train = []
    for batch_idx in tqdm(range(0, len(xs_train), batch_size)):
        x_batch = xs_train[indices[batch_idx : batch_idx + batch_size]]
        y_batch = ys_train[indices[batch_idx : batch_idx + batch_size]]
        # Forward and backward passes
        y_pred = model(x_batch)

        loss = torch.nn.functional.mse_loss(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
    loss_train = sum(loss_train) / len(loss_train)
    print(f"epoch {epoch_idx}: train loss {loss_train}")
    writer.add_scalar("loss_train", loss_train, epoch_idx)

    if epoch_idx % test_step == 0:
        loss_test = []
        for batch_idx in tqdm(range(0, len(xs_test), batch_size)):
            x_batch = xs_test[batch_idx : batch_idx + batch_size]
            y_batch = ys_test[batch_idx : batch_idx + batch_size]
            y_pred = model(x_batch)
            loss = torch.nn.functional.mse_loss(y_pred, y_batch)
            loss_test.append(loss.item())
        loss_test = sum(loss_test) / len(loss_test)
        writer.add_scalar("loss_test", loss_test, epoch_idx)
        print(f"epoch {epoch_idx}: test loss {loss_test}")

    if epoch_idx == max_epochs - 1:
        freqK_base = torch.rand(1).cuda()
        freqK = freqK_base * freq_rate
        sizeK_base = torch.rand(1).cuda()
        sizeK = 1.0 / (1 + sizeK_base * (size_rate - 1))
        vertices = vertices_base * sizeK
        ks = ks_base * freqK / sizeK**0.5
        sample_points_base = torch.rand(point_num_per_sample, 3).cuda()
        rs = (sample_points_base[:, 0] * (bbox_rate - 1) + 1) * size_base * 0.7
        theta = sample_points_base[:, 1] * 2 * np.pi - np.pi
        phi = sample_points_base[:, 2] * np.pi
        xs = rs * torch.sin(phi) * torch.cos(theta)
        ys = rs * torch.sin(phi) * torch.sin(theta)
        zs = rs * torch.cos(phi)
        trg_points = torch.stack([xs, ys, zs], dim=-1)

        x[sample_idx, :, :3] = sample_points_base
        x[sample_idx, :, 3] = sizeK
        x[sample_idx, :, 4] = freqK
        bem_solver = BEM_Solver(vertices, triangles)
        for i in range(mode_num):
            dirichlet_vtx = bem_solver.neumann2dirichlet(ks[i].item(), neumann_vtx[i])
            y[sample_idx, i] = bem_solver.boundary2potential(
                ks[i].item(), neumann_vtx[i], dirichlet_vtx, trg_points
            ).abs()

torch.save(model.state_dict(), f"{data_dir}/model.pt")
