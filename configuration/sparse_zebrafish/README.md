# Sparse zebrafish tracking

This directony contains the configuration files for tracking the dense (green) channel zebrafish from the sparse labeling experiment using SLURM.

This requires a conda environment with the `requirements.txt` file installed and it assumes the environment will be named `ultrack`.

To download the UNet weights use:

```bash
wget https://public.czbiohub.org/royerlab/ultrack/unet_weights/unet-simview.pt
```

The weights are compiled with torchscript, so to load them you can simply:

```python
import torch
model = torch.jit.load('unet-simview.pt')
```

Tracking execution:

```bash
bash main.sh
```
