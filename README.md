# GNNBEM


## Dependencies

CUDA 11.7 (match with the version of PyTorch, which is required by Pytorch CUDA Extension in `src/cuda`):

Pytorch
```
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```
tiny-cuda-nn:
```bash
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
cd tiny-cuda-nn/bindings/torch
python setup.py install
```
Bempp:
```bash
pip install plotly pandas pyopencl[pocl] gmsh
git clone https://github.com/bempp/bempp-cl
cd bempp-cl
python setup.py install
```

Other dependencies:
```bash
pip install numpy scipy numba meshio matplotlib tqdm commentjson protobuf ipywidgets IPython
```


