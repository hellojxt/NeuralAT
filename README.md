# GNNBEM

For bempp & pytorch
```
conda create -n bem python pip -y
conda activate bem
conda install pyopencl pocl=1.3 -c conda-forge -y
pip install plotly gmsh scipy numpy numba ipykernel meshio bempp-cl
cp /etc/OpenCL/vendors/nvidia.icd $(dirname "$(which python)")/../etc/OpenCL/vendors
pip install torch torchvision torchaudio
```


For the installation of fTetWild:
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

For the installation of tiny-cuda-nn:
```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install commentjson
```

Other dependencies:
```bash
pip install matplotlib tqdm
```

