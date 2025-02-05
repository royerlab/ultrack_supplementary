from pathlib import Path
from typing import Optional

import pandas as pd
import napari
import zarr
from ultrack.tracks import get_subgraph
from napari_animation import Animation
from iohub import open_ome_zarr
from skimage.exposure import rescale_intensity,adjust_gamma

FIG_DIR = Path(".")
ROOT_DIR = Path('<DEFINED BY USER>')


def main():
    #%%
    viewer = napari.Viewer()

    #%%
    key = ['A/1/1']

    phase_data_path = ''
    vs_data_path = ''

    vs_mem_segment_ultrack_path = ''
    vs_nuc_segment_ultrack_path = ''
    segment_cp_path = ''

    phase_ds = open_ome_zarr(phase_data_path)
    vs_ds = open_ome_zarr(vs_data_path)
    cp_ds = open_ome_zarr(segment_cp_path)


    scale = phase_ds[key].scale[-2:]
    Z, Y, X = phase_ds[key].data.shape[-3:]

    #Cropping and MIP
    z_slicing = slice(1,5)
    y_slicing = slice(0,Y+1)
    x_slicing = slice(0,X+1)

    phase_arr = phase_ds[key].data[:, 0, 1, y_slicing, x_slicing]
    vs_nuc = vs_ds[key].data[:, 0, z_slicing, y_slicing, x_slicing].max(axis=0)
    vs_mem = vs_ds[key].data[:, 1, z_slicing, y_slicing, x_slicing].max(axis=0)
    cp_arr = cp_ds[key].data[:, 0, 0, y_slicing, x_slicing].astype("uint16")
    vs_mem_ultrack_data= zarr.open(vs_mem_segment_ultrack_path).astype('int')
    vs_nuc_ultrack_data= zarr.open(vs_nuc_segment_ultrack_path).astype('int')


    vs_mem_gamma = 1.0
    vs_nuc_gamma = 1.0
    vs_nuc_clims = (0.0, 25)
    vs_mem_clims = (0.0, 5)
    kwargs = dict(scale=scale, blending="additive")

    #%%
    viewer.add_image(phase_arr,name='Phase', colomap='gray',**kwargs)
    viewer.add_image(vs_nuc,name='vs_nuc', colomap='bop blue',**kwargs)
    viewer.add_image(vs_mem,name='vs_mem', colomap='bop orange',**kwargs)
    viewer.add_labels(vs_nuc_ultrack_data,name='nuc_segment_ult', **kwargs)
    viewer.add_labels(vs_mem_ultrack_data,name='mem_segment_ult', **kwargs)



    #%%

    napari.run()


if __name__= '__main__':
    main()