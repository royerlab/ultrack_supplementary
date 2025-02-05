from pathlib import Path

import numpy as np
import pandas as pd

import napari
from napari.utils import resize_dask_cache

from ultrack.tracks import split_trees
from ultrack.tracks.video import tracks_df_to_3D_video

from utils import remove_multiscale


def main() -> None:
    viewer = napari.Viewer()
    resize_dask_cache(0)

    level = 0
    root = Path("<DEFINED BY USER>")
    # level = 3

    out_path = Path(f"sparse_embryo/videos/level_{level}")
    out_path.mkdir(parents=True, exist_ok=True)

    viewer.open(
        root / "normalized.zarr",
        plugin="napari-ome-zarr",
        rendering="attenuated_mip",
        interpolation3d="nearest",
        attenuation=0.25,
        blending="additive",
    )
    scale = np.asarray(viewer.layers[0].scale[-3:])
    origin = viewer.layers[0].data.shape[-3:] * scale / 2

    remove_multiscale(viewer, level=level)

    viewer.dims.set_point(0, 0)
    viewer.dims.ndisplay = 3
    viewer.camera.perspective = 30.0
    viewer.window.resize(1800, 1200)
    viewer.layers[1].gamma = 0.7

    print("Loading tracks ...")

    tracks = pd.read_csv(root / "analysis/filtered_tracks/tracks_ch1_2024_05_03.csv")
    tracks[["z", "y", "x"]] *= scale
    tracks.sort_values(by=["track_id", "t"], inplace=True)
    tracks["t"] = tracks["t"].astype(int)

    # track_ids = [5652]
    # trees = [tracks[tracks["track_id"].isin(track_ids)].copy()]
    trees = split_trees(tracks)

    viewer.layers.move(1, 0)

    for tree in trees:
        layer = viewer.add_tracks(
            tree[["track_id", "t", "z", "y", "x"]].to_numpy(),
            colormap="hsv",
            blending="opaque",
            name="tree",
        )
        video_path = out_path / f"lineage_{tree['track_id'].iloc[0]}.mp4"

        for l in viewer.layers:
            l._update_thumbnail = lambda *args, **kwargs: None

        t_max = tree["t"].max()
        t_min = tree["t"].min()
        threshold = t_max - 50

        kwargs = {
            t: {
                "zoom": 5.0 if t <= threshold else 1.75,
                "angles": "auto",
                "center": "auto" if t <= threshold else origin
            }
            for t in range(t_min, t_max + 1, 100)
        }
        kwargs[t_max] = {"zoom": 1.75, "center": origin}

        try:
            tracks_df_to_3D_video(
                viewer,
                tree,
                video_path,
                # clipping_planes_layers=[viewer.layers["1"]],
                # clipping_box_size=20,
                time_to_camera_kwargs=kwargs,
                # overwrite=True,
                origin=origin,
            )
        except FileExistsError:
            print(f"Video {video_path} already exists. Skipping ...")

        viewer.layers.remove(layer)


if __name__ == '__main__':
    main()

