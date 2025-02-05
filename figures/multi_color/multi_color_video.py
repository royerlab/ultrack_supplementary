from pathlib import Path

import napari
import zarr
from tifffile import imread
from napari_animation import Animation
from napari.layers.utils.stack_utils import split_rgb

import sys
sys.path.insert(0, "..")

from constants import ULTRACK_DIR
from utils import simple_recording


def main() -> None:

    res_dir = Path("<DEFINE BY USER>")
    data_dir = ULTRACK_DIR / "examples/multi_color_ensemble"
    print(data_dir)

    fig_dir = Path("multi_color")
    fig_dir.mkdir(exist_ok=True)

    # img = imread(img_path)
    img = zarr.open(data_dir / "normalized.zarr")
    scale = (0.75469,) * 3

    viewer = napari.Viewer()
    viewer.window.resize(1800, 1000)

    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"

    layer = viewer.add_image(img, gamma=0.75, scale=scale, rgb=True)
    for l in split_rgb(layer):
        viewer.add_layer(l)
    viewer.layers.remove(layer)

    viewer.add_labels(
        imread(res_dir / "segments.tif"),
        scale=scale,
    ).contour = 4

    viewer.open(
        res_dir / "tracks.csv",
        plugin="ultrack",
        scale=scale,
    )

    simple_recording(viewer, fig_dir / "multi_color.mp4")


if __name__ == "__main__":
    main()
