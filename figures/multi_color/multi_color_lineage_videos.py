from pathlib import Path

import napari
import zarr
import pandas as pd
from tifffile import imread
from napari.layers.utils.stack_utils import split_rgb
from ultrack.tracks.video import tracks_df_to_videos


def main() -> None:

    root = Path("<DEFINE BY USER>")

    fig_dir = Path("./video/lineages")
    fig_dir.mkdir(exist_ok=True)

    # img = imread(img_path)
    img = zarr.open(root / "normalized.zarr")
    scale = (0.75469,) * 2

    viewer = napari.Viewer()
    viewer.window.resize(1600, 1000)

    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"

   # needs PR from https://github.com/napari/napari/pull/6678
    viewer.slice_text.visible = True

    layer = viewer.add_image(img, gamma=0.75, scale=scale, rgb=True)
    for l in split_rgb(layer):
        viewer.add_layer(l)
    viewer.layers.remove(layer)

    # viewer.add_labels(
    #     imread(root / "curated_tracks/segments.tif"),
    #     scale=scale,
    # ).contour = 4

    tracks_path = root / "curated_tracks/tracks.csv"

    layer, = viewer.open(
        tracks_path,
        plugin="ultrack",
        scale=scale,
    )

    viewer.camera.zoom = 4
    df = pd.read_csv(tracks_path)
    df[["y", "x"]] *= scale

    tracks_df_to_videos(
        viewer,
        df,
        fig_dir,
        num_lineages=200,
        sort="length",
        overwrite=True,
        tracks_layer_kwargs=dict(colormap="hsv", tail_length=100, tail_width=2),
    )

if __name__ == "__main__":
    main()
