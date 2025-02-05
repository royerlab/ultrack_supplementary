from pathlib import Path

import napari
import zarr
import pandas as pd
import numpy as np
from napari.layers.utils.stack_utils import split_rgb
from tifffile import imread
from napari_animation import Animation
from ultrack.tracks.graph import get_subgraph
from tqdm import tqdm

import sys
sys.path.insert(0, "..")

from constants import ULTRACK_DIR


def record_cell(
    viewer: napari.Viewer,
    track_df: pd.DataFrame,
    output_path: Path,
    capture_factor: int = 5,
) -> None:

    t_min = track_df["t"].min()

    viewer.dims.set_point(0, t_min)
    viewer.reset_view()
    start_zoom = viewer.camera.zoom
    end_zoom = 4
    zoom = start_zoom
    zoom_factor = 1.2

    animation = Animation(viewer)
    animation.capture_keyframe()
    step = 10

    for t, group in tqdm(track_df.groupby("t")):
        if (t + 1) % step != 0:
            continue

        y, x = group[['y', 'x']].to_numpy()[-1]
        viewer.dims.set_point(0, t)

        zoom = min(zoom * zoom_factor, end_zoom)
        w = (zoom - start_zoom) / (end_zoom - start_zoom)

        new_center = w * np.asarray((0, y, x)) + (1 - w) * np.asarray(viewer.camera.center)
        viewer.camera.center = new_center
        viewer.camera.zoom = zoom

        animation.capture_keyframe(capture_factor * step)

    animation.animate(output_path, fps=60)


def main(track_id: int) -> None:
    res_dir = Path("<DEFINE BY USER>")
    data_dir = ULTRACK_DIR / "examples/multi_color_ensemble"
    print(data_dir)

    fig_dir = Path("multi_color")
    fig_dir.mkdir(exist_ok=True)

    img = zarr.open(data_dir / "normalized.zarr")
    scale = (0.75469,) * 2

    viewer = napari.Viewer()
    viewer.window.resize(1800, 1000)

    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"

    layer = viewer.add_image(img, gamma=0.75, scale=scale, rgb=True)
    for l in split_rgb(layer):
        viewer.add_layer(l)
    
    viewer.layers.remove(layer)

    tracks_df = pd.read_csv(res_dir / "tracks.csv")
    tracks_df[["y", "x"]] *= scale

    subgraph_df = get_subgraph(tracks_df, track_id)
    viewer.add_tracks(subgraph_df[["track_id", "t", "y", "x"]], colormap="twilight", tail_length=50)

    segm = imread(res_dir / "segments.tif")
    not_in = np.isin(segm, subgraph_df["track_id"].unique(), invert=True)
    segm[not_in] = 0

    viewer.add_labels(
        segm,
        scale=scale,
    ).contour = 4

    record_cell(viewer, subgraph_df, fig_dir / f"track_{track_id}.mp4")


if __name__ == "__main__":
    main(472)
