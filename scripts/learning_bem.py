import sys

sys.path.append("./")
import torch
from src.dataset import MeshDataset, TriMesh
import src.network as network
from torch_geometric.loader import DataLoader
import argparse
import commentjson as json
import tinycudann as tcnn
from tqdm import tqdm
from torch_geometric.data import Data
from src.assemble import (
    assemble_single_boundary_matrix,
    assemble_double_boundary_matrix,
)
from torch.utils.tensorboard import SummaryWriter
import os

writer = SummaryWriter()

with open("config/config.json") as f:
    config = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="dataset/ABC_Dataset/surf_mesh")
parser.add_argument("--network", type=str, default="PointNet2")
parser.add_argument("--preprocess", action="store_true")
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--scale_factor", type=float, default=0.0001)

args = parser.parse_args()
scale_factor = args.scale_factor
train_dataset = MeshDataset(args.dataset, "train")
test_dataset = MeshDataset(args.dataset, "test")
if args.preprocess:
    train_dataset.pre_process_meshes()
    test_dataset.pre_process_meshes()
train_dataset.load_pre_processed_mesh()
test_dataset.load_pre_processed_mesh()
train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
)
test_loader = DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

graph_net = getattr(network, args.network)(
    train_dataset[0].x.shape[1], args.hidden_dim
).to(device)

freq_encoder = tcnn.Encoding(1, config["encoding"])
decoder = tcnn.Network(
    freq_encoder.n_output_dims + args.hidden_dim, 1, config["network"]
)


parameters = (
    list(graph_net.parameters())
    + list(freq_encoder.parameters())
    + list(decoder.parameters())
)

optimizer = torch.optim.Adam(parameters, lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
loss_func = torch.nn.SmoothL1Loss()

import matplotlib.pyplot as plt


def plot_surf_data(predict, target):
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(211)
    ax1.plot(predict.detach().cpu().numpy().reshape(-1))
    ax1.set_title("predict")
    ax2 = fig.add_subplot(212)
    ax2.plot(target.detach().cpu().numpy().reshape(-1))
    ax2.set_title("target")
    # set same ylim
    ylim = [
        min(ax1.get_ylim()[0], ax2.get_ylim()[0]),
        max(ax1.get_ylim()[1], ax2.get_ylim()[1]),
    ]
    ax1.set_ylim(ylim)
    ax2.set_ylim(ylim)
    return fig


def step(train=True):
    if train:
        graph_net.train()
        data_loader = train_loader
    else:
        graph_net.eval()
        data_loader = test_loader

    losses = []
    for i, data in enumerate(tqdm(data_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        x_encoded = graph_net(data)
        verts_off = data.vertices_offset
        verts_off_bound = torch.cumsum(verts_off, dim=0)
        loss = 0
        graph_num = data.batch[-1] + 1
        for graph_idx in range(graph_num):
            vertices = (
                data.vertices[
                    verts_off_bound[graph_idx]
                    - verts_off[graph_idx] : verts_off_bound[graph_idx]
                ]
                .contiguous()
                .type(torch.float32)
            )
            mask = data.batch == graph_idx
            triangles = data.triangles[mask].contiguous().type(torch.int32)
            # TriMesh(vertices, triangles).save_obj(f"test_{graph_idx}.obj")
            freq = torch.rand([1, 1], device="cuda")
            wave_number = 5 + (freq * 40).item()
            single_matrix = assemble_single_boundary_matrix(
                vertices, triangles, wave_number
            )
            double_matrix = assemble_double_boundary_matrix(
                vertices, triangles, wave_number
            )
            A = double_matrix - 0.5 * torch.eye(triangles.shape[0], device="cuda")
            b = single_matrix @ data.neumann[mask] / scale_factor
            f_encoded = freq_encoder(freq)
            f_encoded = f_encoded.repeat(triangles.shape[0], 1)
            fused_feats = torch.cat([f_encoded, x_encoded[mask]], axis=1)
            predict = decoder(fused_feats).float()
            loss = loss + loss_func(A @ predict, b)
            if i < 5 and graph_idx == 0:
                fig = plot_surf_data(A @ predict, b)
                writer.add_figure(f"{'train' if train else 'test'}_{i}", fig, epoch)

        loss = loss / args.batch_size
        losses.append(loss.item())
        if train:
            loss.backward()
            optimizer.step()
    return sum(losses) / len(losses)


best_loss = 1e10

for epoch in range(args.epochs):
    train_loss = step(train=True)
    with torch.no_grad():
        test_loss = step(train=False)
    print(f"Epoch {epoch} train loss: {train_loss}, test loss: {test_loss}")
    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("test_loss", test_loss, epoch)
    scheduler.step(test_loss)
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(
            {
                "graph_net": graph_net.state_dict(),
                "freq_encoder": freq_encoder.state_dict(),
                "decoder": decoder.state_dict(),
            },
            f"checkpoints/{args.network}.pth",
        )
