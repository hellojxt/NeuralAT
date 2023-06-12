# GNNBEM

```
conda create -n bem python pip -y
conda activate bem
conda install pyopencl pocl=1.3 -c conda-forge -y
pip install plotly gmsh scipy numpy numba ipykernel meshio bempp-cl
cp /etc/OpenCL/vendors/nvidia.icd $(dirname "$(which python)")/../etc/OpenCL/vendors
pip install torch torchvision torchaudio
```