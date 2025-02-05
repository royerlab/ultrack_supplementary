from pathlib import Path

import numpy as np
import napari
import matplotlib.pyplot as plt
import zarr
from utils import add_scale_bar
from skimage.segmentation import find_boundaries
from skimage.color import label2rgb, gray2rgb

FIG_DIR = Path(".")
DATA_DIR = Path("<DEFINED BY USER>")


def save_img(arr: np.ndarray, path: Path, cmap: str = "gray", **kwargs) -> None:
    fig, ax = plt.subplots()
    ax.imshow(arr, cmap=cmap)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)

    add_scale_bar(
        image_path=path,
        output_path=path,
        pixel_size=0.439,
        bar_length=100,
        **kwargs,
    )



def main() -> None:
    viewer = napari.Viewer()

    layer, = viewer.open(
        "http://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS001_tail.ome.zarr",
        plugin="napari-ome-zarr",
        rendering="attenuated_mip",
        gamma=0.7,
        colormap="magma",
    )
    viewer.window.resize(1800, 1000)
    y = 620 
    x = 620
    slicing = (405, 240) # , slice(y - 150, y + 150), slice(x - 150, x + 150))

    ## save figure using plt
    save_img(layer.data[0][405].max(axis=0), FIG_DIR / "slice_whole.png", cmap="magma")

    img = layer.data[0][slicing]

    save_img(img, FIG_DIR / "slice_crop.png", cmap="magma")

    # demo starts from 400
    slicing = (5, *slicing[1:])

    detection = zarr.open(DATA_DIR / "detection.zarr")

    # det = detection[slicing]
    # print(detection.shape, det.shape, img.shape)
    # det_rgb = label2rgb(det, image=gray2rgb(img), bg_label=0)

    save_img(detection[slicing], FIG_DIR / "slice_detection.png")

    boundaries = zarr.open(DATA_DIR / "boundaries.zarr")
    save_img(boundaries[slicing], FIG_DIR / "slice_boundaries.png", bar_color="black", label_color="black")

    viewer.dims.set_point(0, 409)
    viewer.dims.ndisplay = 3

    viewer.camera.center = (255.6483784580514, 212.9933210941489, 263.6426957897602)
    viewer.camera.angles = (-117.27320704649682, 30.816307934158758, -101.22375932314658)
    viewer.camera.perspective = 30.0

    viewer.screenshot(FIG_DIR / "tail.png") 

    viewer.open(
        DATA_DIR / "tracks.csv",
        plugin="ultrack",
        scale=viewer.layers[0].scale,
        translate=(400, 0, 0, 0),
        blending="opaque",
    )

    viewer.screenshot(FIG_DIR / "tail_tracks.png") 


if __name__ == "__main__":
    main()
