import sys

sys.path.append("./")
from src.net.model import NeuPAT
import torch
from glob import glob
from tqdm import tqdm
import json
import os
import numpy as np


def calculate_helmholtz_loss(xs, ys, zs, ks, sizes, freqs):
    rs = torch.sqrt(xs**2 + ys**2 + zs**2) / bbox_size - 1
    phi = torch.acos(zs / rs) / np.pi
    theta = (torch.atan2(ys, xs) + np.pi) / (2 * np.pi)

    x_batch = torch.stack([rs, theta, phi, sizes, freqs], dim=1).requires_grad_(True)
    predictions = model(x_batch)
    p_real = predictions[:, 0]
    p_imag = predictions[:, 1]

    # Calculate gradients
    grad_p_real = torch.autograd.grad(
        p_real,
        xs,
        grad_outputs=p_real.data.new(p_real.shape).fill_(1),
        create_graph=True,
        allow_unused=True,
    )[0]
    print(grad_p_real)
    grad_p_imag = torch.autograd.grad(
        p_imag,
        inputs,
        grad_outputs=torch.ones_like(p_imag),
        create_graph=True,
        allow_unused=True,
    )[0]

    grad_p_real_x = grad_p_real[:, 0]
    grad_p_real_y = grad_p_real[:, 1]
    grad_p_real_z = grad_p_real[:, 2]

    grad_p_imag_x = grad_p_imag[:, 0]
    grad_p_imag_y = grad_p_imag[:, 1]
    grad_p_imag_z = grad_p_imag[:, 2]

    grad_p_real_xx = torch.autograd.grad(
        grad_p_real_x,
        x,
        grad_outputs=torch.ones_like(grad_p_real_x),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    grad_p_real_yy = torch.autograd.grad(
        grad_p_real_y,
        y,
        grad_outputs=torch.ones_like(grad_p_real_y),
        create_graph=True,
        allow_unused=True,
    )[0]
    grad_p_real_zz = torch.autograd.grad(
        grad_p_real_z,
        z,
        grad_outputs=torch.ones_like(grad_p_real_z),
        create_graph=True,
        allow_unused=True,
    )[0]

    grad_p_imag_xx = torch.autograd.grad(
        grad_p_imag_x,
        x,
        grad_outputs=torch.ones_like(grad_p_imag_x),
        create_graph=True,
        allow_unused=True,
    )[0]
    grad_p_imag_yy = torch.autograd.grad(
        grad_p_imag_y,
        y,
        grad_outputs=torch.ones_like(grad_p_imag_y),
        create_graph=True,
        allow_unused=True,
    )[0]
    grad_p_imag_zz = torch.autograd.grad(
        grad_p_imag_z,
        z,
        grad_outputs=torch.ones_like(grad_p_imag_z),
        create_graph=True,
        allow_unused=True,
    )[0]

    laplacian_p_real = grad_p_real_xx + grad_p_real_yy + grad_p_real_zz
    laplacian_p_imag = grad_p_imag_xx + grad_p_imag_yy + grad_p_imag_zz

    # Ensure k is detached from the computation graph
    k = k.detach()

    helmholtz_loss_real = torch.mean((laplacian_p_real + k**2 * p_real) ** 2)
    helmholtz_loss_imag = torch.mean((laplacian_p_imag + k**2 * p_imag) ** 2)

    helmholtz_loss = helmholtz_loss_real + helmholtz_loss_imag
    return helmholtz_loss


data_dir = "dataset/NeuPAT_new/regular/baseline"
data = torch.load(f"{data_dir}/../data/0.pt")
bbox_size = data["bbox_size"].item()
x = data["x"].cuda()
y = data["y"].cuda()
xx = data["xx"].cuda()

x = x.reshape(-1, x.shape[-1])
y = y.reshape(-1, y.shape[-1])
xx = xx.reshape(-1, xx.shape[-1])

y = y / (y.abs().max() + 1e-6) * 4

xs_train = x[: int(len(x) * 0.8)]
ys_train = y[: int(len(y) * 0.8)]
xs_test = x[int(len(x) * 0.8) :]
ys_test = y[int(len(y) * 0.8) :]
del x, y

with open(f"{data_dir}/net.json", "r") as file:
    net_config = json.load(file)

train_params = net_config.get("train", {})
batch_size = train_params.get("batch_size")

from torch.utils.tensorboard import SummaryWriter

logs = glob(f"{data_dir}/events*")
for log in logs:
    os.remove(log)
writer = SummaryWriter(log_dir=data_dir)


model = NeuPAT(2, net_config).cuda()

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
    xx_start_idx = torch.randint(0, len(xx) - len(xs_train), (1,)).item()
    xx_epoch = xx[xx_start_idx : xx_start_idx + len(xs_train)]

    loss_train = []
    for batch_idx in tqdm(range(0, len(xs_train), batch_size)):
        x_batch = xs_train[indices[batch_idx : batch_idx + batch_size]]
        y_batch = ys_train[indices[batch_idx : batch_idx + batch_size]]
        xx_batch = xx_epoch[batch_idx : batch_idx + batch_size]
        # Forward and backward passes
        y_pred = model(x_batch)
        loss1 = torch.nn.functional.mse_loss(y_pred, y_batch)

        xs = xx_batch[:, 0].requires_grad_(True)
        ys = xx_batch[:, 1].requires_grad_(True)
        zs = xx_batch[:, 2].requires_grad_(True)
        sizes = xx_batch[:, 3]
        freqs = xx_batch[:, 4]
        ks = xx_batch[:, -1]
        loss_reg = calculate_helmholtz_loss(xs, ys, zs, ks, sizes, freqs)

        loss = loss1 + loss_reg
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

    scheduler.step()
    if epoch_idx % 100 == 0:
        torch.save(model.state_dict(), f"{data_dir}/model.pt")
