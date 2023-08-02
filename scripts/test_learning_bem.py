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
from src.linear_solver import BiCGSTAB
import os
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--network", type=str, default="PointNet2")
parser.add_argument("--scale_factor", type=float, default=0.0001)
args = parser.parse_args()
scale_factor = args.scale_factor

if not os.path.exists("images"):
    os.makedirs("images", exist_ok=True)

loss_recorder = LossRecorder(10)
loss_recorder.load(f"checkpoints/{args.network}_{args.epoch}_loss_dict.pth")
from src.fem.visulize import viewer


def plot_surf_data(values, vertices, triangles, tag, image_path):
    vertices = vertices.detach().cpu().numpy()
    triangles = triangles.detach().cpu().numpy()
    values = values.T.detach().cpu().numpy()
    v = viewer(vertices, triangles, values, intensitymode="cell", title=tag)
    v.set_camera(1, 1, 0.1)
    v.save(image_path + tag + ".jpg")
    return image_path + tag + ".jpg"


def test(top_loss, tag):
    idx = 0
    for loss, data in top_loss:
        print(loss)
        vertices = data.vertices.contiguous().type(torch.float32)
        triangles = data.triangles.contiguous().type(torch.int32)
        wave_number = 5 + (data.freq * 40).item()
        double_matrix = assemble_double_boundary_matrix(
            vertices, triangles, wave_number
        )
        single_matrix = assemble_single_boundary_matrix(
            vertices, triangles, wave_number
        )
        A = double_matrix - 0.5 * torch.eye(triangles.shape[0], device="cuda")
        solver = BiCGSTAB(A)
        rhd = single_matrix @ data.neumann / scale_factor
        gt = solver.solve(rhd.reshape(-1)).reshape(-1, 1)
        predict = data.predict
        dir_name = f"images/{tag}/{idx}/"
        os.makedirs(dir_name, exist_ok=True)
        input_path = plot_surf_data(
            data.neumann, vertices, triangles, "Input", dir_name
        )
        gt_path = plot_surf_data(gt, vertices, triangles, "GT", dir_name)
        predict_path = plot_surf_data(predict, vertices, triangles, "Predict", dir_name)
        lhd_path = plot_surf_data(A @ predict, vertices, triangles, "LHD", dir_name)
        rhd_path = plot_surf_data(rhd, vertices, triangles, "RHD", dir_name)
        lhd = A @ predict
        lhd_ = -0.5 * predict
        print(torch.norm(lhd - lhd_) / torch.norm(lhd))
        images = torch.stack(
            [
                torchvision.io.read_image(input_path),
                torchvision.io.read_image(gt_path),
                torchvision.io.read_image(predict_path),
                torchvision.io.read_image(lhd_path),
                torchvision.io.read_image(rhd_path),
            ]
        )
        # make grid
        grid = torchvision.utils.make_grid(images / 255.0, nrow=5)
        torchvision.utils.save_image(grid, f"{dir_name}/grid.jpg")
        idx += 1


test(loss_recorder.top_max_loss, "max loss")
test(loss_recorder.top_min_loss, "min loss")
