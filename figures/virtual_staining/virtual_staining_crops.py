import numpy as np
import matplotlib.pyplot as plt
from iohub import open_ome_zarr
from skimage.exposure import rescale_intensity, adjust_gamma
import napari
from pathlib import Path
from utils import add_scale_bar
from skimage.color import label2rgb, gray2rgb
import zarr
from ultrack import to_tracks_layer, solve, add_new_node, tracks_to_zarr
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
    plt.show()

    if not preview_only:
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
        add_scale_bar(
            image_path=path,
            output_path=path,
            pixel_size=scale,
            bar_length=bar_length,
            **kwargs,
        )


def main():
    root = Path(
        "<DEFINED BY USER>"
    )
    im_path = root / "2-register/2024_02_13_ZIV_DENV-Timelapse_1.zarr"
    vs_path = (
        root / "3-virtual-staining/2DVS/tta/2024_02_13_ZIV_DENV-Timelapse_1_VS_2.zarr"
    )
    cp_path = (
        root
        / "4-segment/tta/2024_02_13_ZIV_DENV-Timelapse_1_VS_tta_mean_labels_cyto2.zarr"
    )

    res_dir = Path("<DEFINED BY USER>")

    # vs_nuc_config_path = "<DEFINED BY USER>/vs_nucleus/config_test.toml"
    # vs_mem_config_path = "<DEFINED BY USER>/vs_membrane/config.toml"

    # vs_mem_foreground_path = "<DEFINED BY USER>/vs_membrane/foreground.zarr"
    # vs_mem_top_arr_path = "<DEFINED BY USER>/vs_membrane/top_arr.zarr"

    key = "A/1/1"
    im_ds = open_ome_zarr(im_path)
    vs_ds = open_ome_zarr(vs_path)
    # cp_ds = open_ome_zarr(cp_path)

    scale = im_ds[key].scale[-2:]
    Z, Y, X = im_ds[key].data.shape[-3:]

    ##########################
    viewer = napari.Viewer()
    viewer.window.resize(1600, 1000)
    viewer.scale_bar.visible = True

    z_slicing = slice(1, 5)
    y_slicing = slice(0, Y)
    x_slicing = slice(0, X)

    phase_arr = im_ds[key].data[:, 0, 2, y_slicing, x_slicing]
    vs_nuc = vs_ds[key].data[:, 0, z_slicing, y_slicing, x_slicing].max(axis=1)
    vs_mem = vs_ds[key].data[:, 1, z_slicing, y_slicing, x_slicing].max(axis=1)
    kwargs = dict(scale=scale, blending="additive")

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

    ctc_l = viewer.open(res_dir / "04_RES", **kwargs, plugin="napari-ctc-io")
    for l in ctc_l:
        l.name = f"trackmate {l.name}"
    
    for l in viewer.layers:
        if isinstance(l, napari.layers.Image):
            viewer._add_layer_from_data(*l.as_layer_data_tuple())

    ctc_l = viewer.open(res_dir / "07_RES", **kwargs, plugin="napari-ctc-io")
    for l in ctc_l:
        l.name = f"ultrack {l.name}"

    # Nuc + track
    # viewer.layers["VS nuclei"].visible = True
    # viewer.layers["VS membrane"].visible = False
    # viewer.layers["Phase"].visible = False

    viewer.camera.zoom = 1.3
    viewer.camera.center = (0.0, 332.31251519153557, 325.3249954812245)

    t = 46
    viewer.dims.set_point(0, t)
    # viewer.screenshot(FIG_DIR / f"nuc_seg_tracks_{t}.png")

    viewer.camera.zoom = 1.3
    viewer.camera.center = (0.0, 332.31251519153557, 325.3249954812245)

    t = 46
    viewer.dims.set_point(0, t)
    # viewer.screenshot(FIG_DIR / f"mem_seg_tracks_{t}.png")

    napari.run()


if __name__ == "__main__":
    main()

