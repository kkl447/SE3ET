# SE3ET CUDA 13 / HKU Setup

## Scope

This note documents the setup flow for running `SE3ET` on a server that only has
CUDA 13 installed, with the HKU RGB reproduction entry at
`experiments/se3eti.hku`.

The original repository targets PyTorch 1.7.1 and CUDA 11.x. That setup is not
practical on an Ada-class GPU host with only CUDA 13 available. The path below
switches to a newer PyTorch/CUDA toolchain while keeping the current HKU
adaptation entry.

## Recommended Environment

- Python 3.10
- PyTorch 2.4.1
- CUDA 12.1 wheel from PyTorch
- CUDA 13 driver/toolkit on host
- Open3D 0.18.0

PyTorch wheels built with CUDA 12.1 can run on a host with a newer NVIDIA
driver. The CUDA extensions are compiled locally with the host toolkit.

## Environment Creation

```bash
conda create -n se3et_cuda13 python=3.10 -y
conda activate se3et_cuda13
```

## Install PyTorch

Use the official CUDA 12.1 wheel index:

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

Validate:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## Install Python Dependencies

Install the repository requirements after PyTorch is already fixed:

```bash
cd ~/kkl_ws/SE3ET
pip install --default-timeout=100 -r requirements.txt
pip install open3d==0.18.0
```

Validate:

```bash
python -c "import torch, open3d as o3d; print(torch.__version__); print(o3d.__version__)"
```

## Build Extensions

Set the compile arch list explicitly for Ada GPUs:

```bash
export TORCH_CUDA_ARCH_LIST="8.9"
```

Build the main extension:

```bash
cd ~/kkl_ws/SE3ET
python setup.py build develop
```

Build `vgtk`:

```bash
cd ~/kkl_ws/SE3ET/geotransformer/modules/e2pn/vgtk
MAX_JOBS=1 python setup.py build_ext -i
```

If the machine has multiple CUDA installations, also set:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## HKU Environment Variables

```bash
export HKU_RGB_DATASET_ROOT=~/kkl_ws/GeoTransformer/data/HKU_MARS/MARS_Dataset_v015_dynamic_s030_rgb
export HKU_METADATA_DIR=metadata_amtown_valtest
export SE3ET_DATASET_TYPE=hku_mars
```

Adjust the dataset root if your server stores data elsewhere.

## Verify HKU Entry

```bash
cd ~/kkl_ws/SE3ET/experiments/se3eti.hku
python test.py --help
python eval_hku.py --help
```

## Run Validation

```bash
python test.py --benchmark=val
python eval_hku.py --benchmark=val --method=lgr
```

## Run Training

```bash
python trainval.py
```

## Notes

- The repository requirements intentionally no longer include `e3nn` or
  `pytorch_memlab` to avoid automatic PyTorch replacement by `pip`.
- `vgtk/setup.py` was updated to compile with C++17, which is required by the
  CUDA 13 toolchain.
- If extension compilation still fails, capture the full `ninja` log and inspect
  the first compile error instead of the final Python traceback.
