import sys

sys.path.append("./")
from src.dataset import MeshDataset


if __name__ == "__main__":
    mesh_dir = "dataset/test_Dataset/surf_mesh"
    dataset = MeshDataset(mesh_dir, "val")
    dataset.pre_process_meshes()
    dataset.load_pre_processed_mesh()
    print(len(dataset))
    print(dataset[0].vertices.shape)
    print(dataset[0].triangles.shape)
    print(dataset[0].adj.shape)
