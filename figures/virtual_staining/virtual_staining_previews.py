# %%
import numpy as np
import matplotlib.pyplot as plt
from iohub import open_ome_zarr
from skimage.exposure import rescale_intensity, adjust_gamma
import napari
import os
from pathlib import Path
from utils import add_scale_bar, contour_overlay
from skimage.color import label2rgb, gray2rgb
from skimage.segmentation import find_boundaries
from scipy.ndimage import gaussian_filter
import zarr
import os
import pandas as pd
from ultrack import to_tracks_layer, tracks_to_zarr
from ultrack.config.config import load_config


FIG_DIR = Path(".")


def save_img(
    arr: np.ndarray,
    path: Path,
    scale,
    preview_only=False,
    cmap: str = "gray",
    bar_length=100,
    **kwargs,
) -> None:
    fig, ax = plt.subplots()
    ax.imshow(arr, cmap=cmap)
    ax.axis("off")
    fig.tight_layout()
    # plt.show()

    if not preview_only:
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
        # add_scale_bar(
        #     image_path=path,
        #     output_path=path,
        #     pixel_size=scale,
        #     bar_length=bar_length,
        #     **kwargs,
        # )
    else:
        print("PREVIEW MODE, NOT SAVING", path)

    # plt.close()


# %%
def main():
    # %%
    root = Path("<DEFINED BY USER>")
    im_path = root / "2-register/2024_02_13_ZIV_DENV-Timelapse_1.zarr"
    vs_path = (
        root / "3-virtual-staining/2DVS/tta/2024_02_13_ZIV_DENV-Timelapse_1_VS_2.zarr"
    )
    cp_path = (
        root
        / "4-segment/tta/2024_02_13_ZIV_DENV-Timelapse_1_VS_tta_mean_labels_cyto2.zarr"
    )

    vs_nuc_config_path = "<DEFINED BY USER>/vs_nucleus/config_test.toml"
    vs_mem_config_path = "<DEFINED BY USER>/vs_membrane/config.toml"

    vs_mem_foreground_path = "<DEFINED BY USER>/vs_membrane/foreground.zarr"
    vs_mem_top_arr_path = "<DEFINED BY USER>/vs_membrane/top_arr.zarr"
    cp_phase_arr_path = "<DEFINED BY USER>/cellpose/data/test_phase_A_1_1.zarr"
    key = "A/1/1"
    # %%
    im_ds = open_ome_zarr(im_path)
    vs_ds = open_ome_zarr(vs_path)
    cp_ds = open_ome_zarr(cp_path)
    cp_phase_ds = open_ome_zarr(cp_phase_arr_path)
    contour_radius = 9

    scale = im_ds[key].scale[-2:]
    Z, Y, X = im_ds[key].data.shape[-3:]

    # %%
    # Cropping the FOV to some cells
    t_idx = 29
    z_slicing = slice(1, 5)
    y_slicing = slice(0, Y)
    x_slicing = slice(0, X)
    # %%
    phase_arr = im_ds[key].data[t_idx, 0, 1, y_slicing, x_slicing]
    vs_nuc = vs_ds[key].data[t_idx, 0, z_slicing, y_slicing, x_slicing].max(axis=0)
    vs_mem = vs_ds[key].data[t_idx, 1, z_slicing, y_slicing, x_slicing].max(axis=0)
    cp_nuc_arr = cp_ds[key].data[t_idx, 0, 0, y_slicing, x_slicing].astype("uint16")
    cp_mem_arr = cp_ds[key].data[t_idx, 1, 0, y_slicing, x_slicing].astype("uint16")
    cp_phase_arr = cp_phase_ds[key].data[t_idx, 0, 0, y_slicing, x_slicing].astype("uint16")
 
    # %%
    # VS Composite
    colormap_1 = [0.1254902, 0.6784314, 0.972549]  # bop blue
    colormap_2 = [0.972549, 0.6784314, 0.1254902]  # bop orange
    vs_mem_gamma = 1.0
    vs_nuc_gamma = 1.0
    vs_nuc_clims = (0.0, 25)
    vs_mem_clims = (0.0, 5)

    vs_nuc = rescale_intensity(vs_nuc, in_range=tuple(vs_nuc_clims))
    vs_mem = rescale_intensity(vs_mem, in_range=tuple(vs_mem_clims))
    vs_mem = adjust_gamma(vs_mem, gamma=vs_mem_gamma)
    vs_nuc = rescale_intensity(vs_nuc, out_range=(0, 1))
    vs_mem = rescale_intensity(vs_mem, out_range=(0, 1))
    vs_nuc = adjust_gamma(vs_nuc, gamma=vs_nuc_gamma)

    vs_composite_arr = np.zeros((*vs_nuc.shape[-2:], 3))
    vs_composite_arr[:, :, 0] = vs_nuc * colormap_1[0] + vs_mem * colormap_2[0]
    vs_composite_arr[:, :, 1] = vs_nuc * colormap_1[1] + vs_mem * colormap_2[1]
    vs_composite_arr[:, :, 2] = vs_nuc * colormap_1[2] + vs_mem * colormap_2[2]

    # Save the RGB images
    save_img(
        vs_composite_arr,
        path=FIG_DIR / "vs_nuc_mem_composite.png",
        preview_only=False,
        cmap=None,
        scale=scale[-1],
        bar_length=100,
    )
    # %%
    # Phase Composite
    phase_clims = (-0.088, 0.088)
    phase = rescale_intensity(phase_arr, in_range=phase_clims)
    # Save the Phase image
    save_img(
        phase,
        path=FIG_DIR / "phase_whole_fov.png",
        preview_only=False,
        cmap="gray",
        scale=scale[-1],
        bar_length=100,
    )

    # CP Phase
    phase -= phase.min()
    phase /= phase.max()
    phase *= 255

    cp_mem = contour_overlay(phase, cp_phase_arr.copy(), radius=contour_radius)
    save_img(
        cp_mem,
        path=FIG_DIR / "cp_phase.png",
        preview_only=False,
        cmap=None,
        scale=scale[-1],
        bar_length=100,
    )

    # %%
    # Cellpose Segmentations
    cp_mem = label2rgb(cp_mem_arr)
    save_img(
        cp_mem,
        path=FIG_DIR / "cp_mem.png",
        preview_only=False,
        cmap=None,
        scale=scale[-1],
        bar_length=100,
    )

    cp_nuc = label2rgb(cp_nuc_arr)
    save_img(
        cp_nuc,
        path=FIG_DIR / "cp_nuc.png",
        preview_only=False,
        cmap=None,
        scale=scale[-1],
        bar_length=100,
    )

    contour = find_boundaries(cp_phase_arr, mode="both")
    contour = gaussian_filter(contour.astype("float32"), sigma=4.0)
    contour /= contour.max()
    save_img(
        contour,
        path=FIG_DIR / "cp_phase_contour.png",
        preview_only=False,
        cmap="gray",
        scale=scale[-1],
        bar_length=100,
    )

    # %%
    # VS Nuc ultrack segment
    vs_nuc_ult_config = load_config(vs_nuc_config_path)
    tracks_df, nuc_ult_graph = to_tracks_layer(vs_nuc_ult_config)
    tracks_df = tracks_df.sort_values(["track_id", "t"])
    vs_nuc_segms = tracks_to_zarr(vs_nuc_ult_config, tracks_df)
    segment_tidx = label2rgb(vs_nuc_segms[t_idx])
    save_img(
        segment_tidx,
        path=FIG_DIR / f"vs_nuc_ultrack_segment_t0.png",
        preview_only=False,
        cmap=None,
        scale=scale[-1],
        bar_length=100,
    )

    # %%
    # VS mem foreground
    vs_mem_foreground = zarr.open(vs_mem_foreground_path)
    vs_mem_foreground_t_idx = vs_mem_foreground[t_idx]

    save_img(
        vs_mem_foreground_t_idx,
        path=FIG_DIR / f"vs_mem_foreground0.png",
        preview_only=False,
        cmap="gray_r",
        scale=scale[-1],
        bar_length=100,
    )
    # %%
    # VS mesegmmop countour
    vs_mem_segm_arr = zarr.open(vs_mem_top_arr_path)
    vs_mem_segm_arr_t_idx = vs_mem_segm_arr[t_idx]
    save_img(
        vs_mem_segm_arr_t_idx,
        path=FIG_DIR / f"vs_mem_top_arr0.png",
        preview_only=False,
        cmap="gray",
        scale=scale[-1],
        bar_length=100,
    )
    # napari.view_image(vs_mem_segm_arr)
    # napari.run()

    # %%
    # VS Mem ultrack segment
    vs_mem_ult_config = load_config(vs_mem_config_path)
    tracks_df, mem_ult_graph = to_tracks_layer(vs_mem_ult_config)
    tracks_df = tracks_df.sort_values(["track_id", "t"])
    vs_mem_segms = tracks_to_zarr(vs_mem_ult_config, tracks_df)
    segment_tidx = vs_mem_segms[t_idx]

    vs_mem_segm_arr_t_idx -= vs_mem_segm_arr_t_idx.min()
    vs_mem_segm_arr_t_idx /= vs_mem_segm_arr_t_idx.max()
    vs_mem_segm_arr_t_idx *= 255

    segment_tidx = contour_overlay(vs_mem_segm_arr_t_idx, segment_tidx, radius=contour_radius)

    save_img(
        segment_tidx,
        path=FIG_DIR / f"vs_mem_ultrack_segment_t0.png",
        preview_only=False,
        cmap=None,
        scale=scale[-1],
        bar_length=100,
    )
    ##########################
    # %%
    viewer = napari.Viewer()
    # %%
    z_slicing = slice(1, 5)
    y_slicing = slice(0, Y)
    x_slicing = slice(0, X)

    phase_arr = im_ds[key].data[:, 0, 2, y_slicing, x_slicing]
    vs_nuc = vs_ds[key].data[:, 0, z_slicing, y_slicing, x_slicing].max(axis=1)
    vs_mem = vs_ds[key].data[:, 1, z_slicing, y_slicing, x_slicing].max(axis=1)
    kwargs = dict(scale=scale, blending="additive")

    # %%
    vs_mem_clims = (0, 8)
    vs_nuc_clims = (0, 44)
    phase_clims = (-0.088, 0.088)

    viewer.add_image(phase_arr, name="Phase", contrast_limits=phase_clims, **kwargs)
    viewer.add_image(
        vs_nuc,
        name="VS nuclei",
        colormap="bop blue",
        contrast_limits=vs_nuc_clims,
        **kwargs,
    )
    viewer.add_image(
        vs_mem,
        name="VS membrane",
        colormap="bop orange",
        contrast_limits=vs_mem_clims,
        **kwargs,
    )
    # viewer.add_labels(cp_arr, name="Cellpose", **kwargs)
    # viewer.add_image(top_arr, name="EDT", colormap="magma", **kwargs)
    # viewer.add_labels(fg_arr, name="Foreground", scale=scale)
    viewer.add_image(vs_mem_segm_arr, name="VS TOP", colormap="gray", **kwargs, visible=False)

    # %%
    # Add the vs nuc ultrack segments
    viewer.add_labels(vs_nuc_segms, name="NUC_SEGM", visible=False, scale=scale).contour = 9
    viewer.add_tracks(
        tracks_df[["track_id", "t", "y", "x"]],
        # graph=nuc_ult_graph,
        blending="opaque",
        colormap="hsv",
        tail_width=2,
        name="TRACKS_NUC_ULT",
        tail_length=10,
        scale=scale,
    )
    viewer.add_labels(vs_mem_segms, name="MEM_SEGM", visible=False, scale=scale).contour = 9
    viewer.add_tracks(
        tracks_df[["track_id", "t", "y", "x"]],
        # graph=mem_ult_graph,
        blending="opaque",
        colormap="hsv",
        tail_width=2,
        name="TRACKS_MEM_ULT",
        tail_length=10,
        scale=scale,
    )

    viewer.window.resize(1600, 1000)
    viewer.dims.set_point(0, t_idx)

    viewer.camera.zoom = 1.3
    viewer.camera.center = (0.0, 332.31251519153557, 325.3249954812245)

    # %%
    # Nuc + track
    viewer.layers["Phase"].visible = True
    viewer.layers["VS membrane"].visible = False
    viewer.layers["VS nuclei"].visible = True
    viewer.layers["NUC_SEGM"].visible = True
    viewer.layers["TRACKS_NUC_ULT"].visible = True

    viewer.screenshot(FIG_DIR / f"nuc_seg_tracks_0.png")

    viewer.layers["VS nuclei"].visible = False
    viewer.layers["NUC_SEGM"].visible = False
    viewer.layers["TRACKS_NUC_ULT"].visible = False

    # %%
    viewer.layers["MEM_SEGM"].visible = True
    viewer.layers["TRACKS_MEM_ULT"].visible = True

    viewer.screenshot(FIG_DIR / f"mem_seg_tracks_0.png")

    viewer.layers["MEM_SEGM"].visible = False
    viewer.layers["TRACKS_MEM_ULT"].visible = False

    # %%
    ctc_dir = Path("<DEFINED BY USER>/07_RES")
    viewer.open(ctc_dir, plugin="napari-ctc-io", scale=scale, tail_length=10, tail_width=2)
    viewer.layers["labels"].contour = 9
    viewer.layers["tracks"].display_graph = False
    viewer.screenshot(FIG_DIR / f"phase_seg_tracks_0.png")

    # %%
    napari.run()


# %%
if __name__ == "__main__":
    main()

# %%
