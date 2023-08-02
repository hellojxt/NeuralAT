import sys

sys.path.append("./")
import torch
from src.dataset import MeshDataset, TriMesh
import src.network as network
from src.utils import LossRecorder
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
import torchvision

writer = SummaryWriter()

with open("config/config.json") as f:
    config = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="dataset/ABC_Dataset/surf_mesh_remeshed_neumann"
)
parser.add_argument("--network", type=str, default="PointNet2")
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--scale_factor", type=float, default=0.0001)

args = parser.parse_args()
scale_factor = args.scale_factor
train_dataset = MeshDataset(args.dataset, "train")
test_dataset = MeshDataset(args.dataset, "test")
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

if not os.path.exists("images"):
    os.makedirs("images", exist_ok=True)

from src.fem.visulize import viewer


def plot_surf_data(values, vertices, triangles, tag):
    image_path = f"images/{epoch}"
    vertices = vertices.detach().cpu().numpy()
    triangles = triangles.detach().cpu().numpy()
    values = values.T.detach().cpu().numpy()
    v = viewer(vertices, triangles, values, intensitymode="cell", title=tag)
    v.set_camera(1, 1, 0.1)
    v.save(image_path + tag + ".jpg")
    return image_path + tag + ".jpg"


log_step_num = 2


def step(train=True):
    if train:
        graph_net.train()
        data_loader = train_loader
    else:
        graph_net.eval()
        data_loader = test_loader

    losses = []
    loss_recorder = LossRecorder(10)
    for i, data in enumerate(tqdm(data_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        x_encoded = graph_net(data)
        verts_off = data.vertices_offset
        verts_off_bound = torch.cumsum(verts_off, dim=0)
        loss = 0
        graph_num = data.batch[-1] + 1
        for graph_idx in range(graph_num):
            # clear the cache
            torch.cuda.empty_cache()
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
            freq = data.freq[graph_idx].reshape(1, 1)
            wave_number = 5 + (freq * 40).item()
            if i < log_step_num:
                test_indices = torch.arange(triangles.shape[0], device="cuda").int()
            else:
                test_indices = torch.randint(
                    0, triangles.shape[0], (64,), device="cuda"
                ).int()
            single_matrix = assemble_single_boundary_matrix(
                vertices, triangles, wave_number, test_indices
            )
            double_matrix = assemble_double_boundary_matrix(
                vertices, triangles, wave_number, test_indices
            )
            f_encoded = freq_encoder(freq)
            f_encoded = f_encoded.repeat(triangles.shape[0], 1)
            fused_feats = torch.cat([f_encoded, x_encoded[mask]], axis=1)
            predict = decoder(fused_feats).float()
            lhd = double_matrix @ predict - 0.5 * predict[test_indices]
            rhd = single_matrix @ data.neumann[mask] / scale_factor
            loss_one = loss_func(lhd, rhd)
            loss = loss + loss_one
            if i < log_step_num and graph_idx == 0:
                lhd_path = plot_surf_data(lhd, vertices, triangles, "LHD")
                rhd_path = plot_surf_data(rhd, vertices, triangles, "RHD")
                input_path = plot_surf_data(
                    data.neumann[mask], vertices, triangles, "Input"
                )
                predict_path = plot_surf_data(predict, vertices, triangles, "Predict")
                images = torch.stack(
                    [
                        torchvision.io.read_image(lhd_path),
                        torchvision.io.read_image(rhd_path),
                        torchvision.io.read_image(input_path),
                        torchvision.io.read_image(predict_path),
                    ]
                )
                writer.add_images(
                    f"{'train' if train else 'test'}_{i}",
                    images,
                    epoch,
                    dataformats="NCHW",
                )
            if train:
                data_one = Data(
                    vertices=vertices,
                    triangles=triangles,
                    neumann=data.neumann[mask],
                    predict=predict,
                    freq=freq,
                )
                loss_recorder.update(loss_one.item(), data_one)

        loss = loss / graph_num
        losses.append(loss.item())
        if train:
            loss.backward()
            optimizer.step()
    if train and epoch % 10 == 0:
        loss_recorder.save(f"checkpoints/{args.network}_{epoch}_loss_dict.pth")
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
