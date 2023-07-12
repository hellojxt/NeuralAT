# GNNBEM


## Dependencies

Pytorch
```
pip install torch torchvision torchaudio
```
PyG
```
pip install torch_geometric
# Optional dependencies:
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```
tiny-cuda-nn (modified for 1 dimension):
```bash
pip install git+https://github.com/hellojxt/tiny-cuda-nn/#subdirectory=bindings/torch
pip install commentjson
```

Other dependencies:
```bash
pip install matplotlib tqdm
```

### Optional dependencies:
bempp (required for `scripts/test_cuda.py`):
```bash
conda create -n bem python pip -y
conda activate bem
conda install pyopencl pocl=1.3 -c conda-forge -y
pip install plotly gmsh scipy numpy numba ipykernel meshio bempp-cl
cp /etc/OpenCL/vendors/nvidia.icd $(dirname "$(which python)")/../etc/OpenCL/vendors
```

fTetWild (used for remesh):
```bash
sudo apt-get install libgmp-dev
git clone https://github.com/wildmeshing/fTetWild.git
cd fTetWild
mkdir build
cd build
cmake ..
make
```
Then add the build directory to the environment variable PATH.


## Usage

1. Prepare the data. Put triangle meshed (.obj) in `dataset/ABC_Dataset/surf_mesh`
2. Traing the model
```bash
python scripts/learning_bem.py
```

