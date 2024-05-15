import sys

sys.path.append("./")
from src.net.esm import NeuralExpansionAT
import torch
from glob import glob
import torch
from tqdm import tqdm
import json
import os

data_dir = "dataset/NeuPAT_new/scale/baseline"
data_points_lst = glob(f"{data_dir}/../data/*.pt")
xs = []
ys = []
for data_points in data_points_lst:
    data = torch.load(data_points)
    x = data["x"].reshape(-1, 5)
    y = data["y"].reshape(-1, 1)
    xs.append(x)
    ys.append(y)

xs = torch.cat(xs, dim=0)
ys = torch.cat(ys, dim=0)
ys = ys / ys.max()

xs_train = xs[: int(len(xs) * 0.9)].cuda()
ys_train = ys[: int(len(ys) * 0.9)].cuda()
xs_test = xs[int(len(xs) * 0.9) :].cuda()
ys_test = ys[int(len(ys) * 0.9) :].cuda()
del xs, ys

with open(f"{data_dir}/net.json", "r") as file:
    net_config = json.load(file)

with open(f"{data_dir}/../config.json", "r") as file:
    data_config = json.load(file)

train_params = net_config.get("train", {})
batch_size = train_params.get("batch_size")

from torch.utils.tensorboard import SummaryWriter

logs = glob(f"{data_dir}/events*")
for log in logs:
    os.remove(log)
writer = SummaryWriter(log_dir=data_dir)


model = NeuralExpansionAT(net_config, data_config).cuda()

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
        y_pred = ((y_pred + 10e-6) / 10e-6).log10()
        y_batch = ((y_batch + 10e-6) / 10e-6).log10()
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
            loss = torch.nn.functional.l1_loss(y_pred, y_batch)
            loss_test.append(loss.item())
        loss_test = sum(loss_test) / len(loss_test)
        writer.add_scalar("loss_test", loss_test, epoch_idx)
        print(f"epoch {epoch_idx}: test loss {loss_test}")

torch.save(model.state_dict(), f"{data_dir}/model.pt")
