# %%

import  time

import napari
import numpy as np
import cupy as cp
import cucim.skimage.morphology as morph
from numpy.typing import ArrayLike
from iohub import open_ome_zarr
from tqdm import tqdm
from rich import print
from ultrack import Tracker, MainConfig
from ultrack.imgproc import robust_invert, normalize
from ultrack.utils.array import array_apply, create_zarr
import toml
import zarr

# GLOBAL for debugging
CROP = False

def line_closing(arr: np.ndarray, size: int) -> np.ndarray:
    lines = [
        cp.ones((size, 1), dtype=bool),
        cp.ones((1, size), dtype=bool),
        cp.eye(size, dtype=bool),
        cp.eye(size, dtype=bool)[:, ::-1],
    ]
    closed = cp.stack([morph.closing(arr, l) for l in lines]).max(axis=0)
    return closed


def detect_foreground(
    mem: ArrayLike,
    threshold: float,
    disk=(7,3),
    area_threshold=500,
) -> np.ndarray:
    mem = cp.asarray(mem)
    fg = cp.zeros_like(mem, dtype=bool)
    big_disk = morph.disk(disk[0])
    small_disk = morph.disk(disk[1])
    for t in tqdm(range(mem.shape[0]), "Processing foreground"):
        mask = morph.closing(mem[t], big_disk) > threshold
        mask = morph.remove_small_objects(mask, area_threshold)
        mask = morph.remove_small_holes(mask, area_threshold)
        mask = morph.erosion(mask, small_disk)
        fg[t] = mask
    return fg.get()


def normalize_add(
    nuc: np.ndarray,
    mem: np.ndarray,
)-> np.ndarray:
    nuc = cp.asarray(nuc)
    mem = cp.asarray(mem)
    nuc= cp.clip(nuc,0,nuc.max())
    mem=cp.clip(mem,0,mem.max())
    norm_mem = normalize(mem, gamma=0.4)
    norm_nuc = normalize(nuc,gamma=1)
    sum_arr = cp.add(norm_nuc,norm_mem)
    return sum_arr.get()

def combine_nuc_mem(
    nuc: np.ndarray,
    mem: np.ndarray,
    sigma: float,
    weight: float,
    line_size: int,
    gamma:float=1,
) -> np.ndarray:
    nuc = cp.asarray(nuc)
    mem = cp.asarray(mem)
    mem = line_closing(mem, line_size)
    inv_nuc = cp.asarray(robust_invert(nuc, sigma=sigma))
    norm_mem = normalize(mem, gamma=gamma)
    top = (norm_mem + weight * inv_nuc) / (1 + weight)
    return top.get()


# %%
def main() -> None:
    # OPTIONAL, but recommended:
    # replace with the downloaded dataset path
    dataset_path = "/hpc/websites/public.czbiohub.org/royerlab/ultrack/a549_virtual_staining.ome.zarr"

    key = "A/1/1"
    z_slicing = slice(1,5)
    dataset = open_ome_zarr(dataset_path)
    scale = dataset[key].scale[-2:]
    channel_names = dataset[key].channel_names
    vs_mem_channel_idx = channel_names.index("Membrane_prediction")
    vs_nuc_channel_idx = channel_names.index("Nuclei_prediction")
    phase_channel_idx = channel_names.index("Phase3D")

    # Cropping the FOV to some cells
    if CROP:
        top_left_coords = (int(240 / scale[-2]), int(182 / scale[-1]))
        fov_size = int(400 / scale[-2])
        y_slicing = slice(top_left_coords[-2], top_left_coords[-2] + fov_size)
        x_slicing = slice(top_left_coords[-1], top_left_coords[-1] + fov_size)
    else:
        y_slicing = slice(None)
        x_slicing = slice(None)
    print(f"slicing y: {y_slicing}, x: {x_slicing}")
    im_arr = dataset[key].data[:, phase_channel_idx, (z_slicing.stop-z_slicing.start)//2, y_slicing, x_slicing]
    mem_arr = dataset[key].data[:, vs_mem_channel_idx, z_slicing, y_slicing, x_slicing].max(axis=1)
    nuc_arr = dataset[key].data[:, vs_nuc_channel_idx, z_slicing, y_slicing, x_slicing].max(axis=1)
    # %%
    nuc_mem_arr = create_zarr(shape=nuc_arr.shape, dtype=np.float32)
    array_apply(
        nuc_arr,
        mem_arr,
        out_array=nuc_mem_arr,
        func=normalize_add
    )
    fg_arr = detect_foreground(nuc_mem_arr, threshold=0.15,disk=(7,3),area_threshold=1_000)
    top_arr = create_zarr(shape=im_arr.shape, dtype=np.float32)
    array_apply(
        nuc_arr,
        mem_arr,
        out_array=top_arr,
        func=combine_nuc_mem,
        sigma=25.0,
        weight=0.5,
        line_size=10,
        gamma=1.0
    )
    # %%
    # Visualization
    viewer = napari.Viewer()
    kwargs = dict(scale=scale, blending="additive")

    viewer.add_image(im_arr, name="Phase", colormap="gray", **kwargs)
    viewer.add_image(nuc_arr, name="VS nuclei", colormap="green", **kwargs)
    viewer.add_image(mem_arr, name="VS membrane", colormap="magenta", **kwargs)
    viewer.add_image(top_arr, name="Combined", colormap="magma", **kwargs)
    viewer.add_labels(fg_arr, name="Foreground", scale=scale)
    #%%
    # Tracking configuration
    cfg = MainConfig()

    # NOTE: working_dir is used to save the results
    cfg.data_config.working_dir = "./"

    cfg.segmentation_config.min_area = 1_500 
    cfg.segmentation_config.max_area = 50_000 
    cfg.segmentation_config.n_workers = 15

    cfg.linking_config.n_workers = 10
    cfg.linking_config.max_distance =25 
    cfg.linking_config.max_neighbors =5 

    cfg.tracking_config.disappear_weight = -2 
    cfg.tracking_config.appear_weight = -0.001 
    cfg.tracking_config.division_weight = -0.001 

    print(cfg)

    start_t = time.time()
    tracker = Tracker(cfg)
    tracker.track(
        detection=fg_arr,
        edges=top_arr,
        scale=scale,
        overwrite=True,
    )
    print(f"Tracking took {time.time()-start_t:.3f} s")

    tracks_df, graph = tracker.to_napari()
    tracks_path = './tracks.csv'
    tracks_df.to_csv(tracks_path, index=False)
    segments = tracker.to_zarr(tracks_df=tracks_df, store_or_path='./segmentation.zarr',overwrite=True)

    viewer.add_labels(segments, name="Segments", scale=scale)

    viewer.add_tracks(
        tracks_df[["track_id", "t", "y", "x"]], graph=graph, name="Tracks", scale=scale,blending='opaque',colormap='hsv'
    )
    
    # Save the foreground
    z = zarr.array(fg_arr,dtype='float32')
    zarr.save('./foreground.zarr',z)
    z = zarr.array(segments,dtype='int32')
    zarr.save('./segmentations.zarr',z)

    # Save the config
    with open('./config.toml', mode="w") as f:
        toml.dump(cfg.dict(by_alias=True), f)

    napari.run()
# %%
if __name__ == "__main__":
    main()

# %%
