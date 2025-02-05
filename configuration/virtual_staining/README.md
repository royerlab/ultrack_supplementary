# Tracking of label-free images by leveraging virtual staining

This directory contains the configuration files for 

For information regarding the virtual staining of landmark organelles from label-free images, please refer to the [paper](https://www.biorxiv.org/content/10.1101/2024.05.31.596901) and the [repository](https://github.com/mehta-lab/VisCy).

To run this experiment, you need to have a conda environment with the `requirements.txt` file installed and it assumes the environment will be named `ultrack`.

To download the dataset:
```bash
wget https://public.czbiohub.org/royerlab/ultrack/a549_virtual_staining.ome.zarr
```

To execute the tracking ensure to replace the paths in `/vs_membrane/main.py`
```bash
python /vs_membrane/main.py
```
