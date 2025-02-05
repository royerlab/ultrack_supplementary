from pathlib import Path

import napari
import zarr
import pandas as pd
from napari.layers.utils.stack_utils import split_rgb
from ultrack.tracks.video import tracks_df_to_moving_2D_plane_video


def main() -> None:

    root = Path("<DEFINE BY USER>")

    fig_dir = Path("multi_color")
    fig_dir.mkdir(exist_ok=True)

    # img = imread(img_path)
    img = zarr.open(root / "normalized.zarr")
    scale = (0.75469,) * 2

    viewer = napari.Viewer()
    viewer.window.resize(1600, 1000)

    viewer.scale_bar.visible = False
    viewer.scale_bar.unit = "um"

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

    viewer.dims.ndisplay = 3
    # viewer.camera.center = (430, 350, 800)
    # viewer.camera.zoom = 0.45
    # viewer.camera.angles = (-165, 35, 45)

    viewer.camera.center = (725, 445, 800)
    viewer.camera.zoom = 0.45
    viewer.camera.angles = (-25, 35, -35)

    # napari.run(); return

    graph = layer.graph
    viewer.layers.remove(layer)

    df = pd.read_csv(tracks_path)
    video_path = fig_dir / "video/multi_color_moving_slice.mp4"
    video_path.parent.mkdir(exist_ok=True, parents=True)

    # simple_recording(viewer, fig_dir / "multi_color.mp4")
    tracks_df_to_moving_2D_plane_video(
        viewer,
        df,
        video_path,
        plane_mov_scale=3,
        tracks_layer_kwargs=dict(
            blending="translucent",  # "opaque", maybe
            colormap="hsv",
            name="tracks",
            scale=scale,
            opacity=0.75,
            tail_length=300,
            tail_width=1,
            graph=graph,
        ),
        overwrite=True,
    )


if __name__ == "__main__":
    main()
