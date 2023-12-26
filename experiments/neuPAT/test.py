import sys

sys.path.append("./")
from src.net.model import NeuPAT
import torch
from glob import glob
import torch
from tqdm import tqdm
import json
from src.timer import Timer

data_dir = sys.argv[1]
with open(f"{data_dir}/config.json", "r") as file:
    config_data = json.load(file)
data_points_lst = glob(f"{data_dir}/data_*.pt")
xs = []
ys = []
for data_points in data_points_lst:
    data = torch.load(data_points)
    xs.append(data["x"])
    ys.append(data["y"])

xs = torch.cat(xs, dim=0).reshape(-1, xs[0].shape[-1])
ys = torch.cat(ys, dim=0).reshape(-1, ys[0].shape[-1])
ys = ((ys + 10e-6) / 10e-6).log10()

xs = xs.cuda()
ys = ys.cuda()
train_params = config_data.get("train", {})
batch_size = train_params.get("batch_size")

model = NeuPAT(
    n_output_dims=ys.shape[-1],
    src_move=True,
    trg_move=True,
    src_rot=True,
    freq_num=0,
    n_neurons=train_params.get("n_neurons"),
    n_hidden_layers=train_params.get("n_hidden_layers"),
).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=train_params.get("lr"))
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=train_params.get("step_size"), gamma=train_params.get("gamma")
)


# Custom shuffle function that operates on the GPU
def shuffle_tensors(x, y):
    indices = torch.randperm(x.size(0)).cuda()
    return x[indices], y[indices]


max_epochs = train_params.get("max_epochs")

for epoch_idx in tqdm(range(max_epochs)):
    # Shuffle data
    xs_shuffled, ys_shuffled = shuffle_tensors(xs, ys)

    # Create batches manually
    for batch_idx in range(0, len(xs_shuffled), batch_size):
        x_batch = xs_shuffled[batch_idx : batch_idx + batch_size]
        y_batch = ys_shuffled[batch_idx : batch_idx + batch_size]

        # Forward and backward passes
        y_pred = model(x_batch)
        loss = torch.nn.functional.mse_loss(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch_idx}: {loss.item()}")

torch.save(model.state_dict(), f"{data_dir}/model_{sys.argv[2]}.pt")
