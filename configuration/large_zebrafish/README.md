# Large zebrafish tracking

This directony contains the configuration files for tracking the large zebrafish using SLURM.

This requires a conda environment with the `requirements.txt` file installed and it assumes the environment will be named `ultrack`.

To download the UNet weights use:

```bash
wget https://public.czbiohub.org/royerlab/ultrack/unet_weights/unet-daxi.pt
```

The weights are compiled with torchscript, so to load them you can simply:

```python
import torch
model = torch.jit.load('unet-daxi.pt')
```

Tracking execution:

```bash
bash main.sh
```

