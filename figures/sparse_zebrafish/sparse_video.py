from pathlib import Path
import napari
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from napari.utils import resize_dask_cache
from skimage.segmentation import relabel_sequential
from napari_animation import Animation
from napari.layers import Tracks


FIG_DIR = Path(".")
FIG_DIR.mkdir(exist_ok=True)


def load_tracks_layers(
    viewer: napari.Viewer,
    path: Path,
    *,
    start_time: int = 0,
    **kwargs,
) -> Tracks:

    df = pd.read_csv(path)
    df["track_id"], _, _ = relabel_sequential(df["track_id"].to_numpy(dtype=int))

    df = df.query(f"t >= {start_time}")

    layer = viewer.add_tracks(df[["track_id", "t", "z", "y", "x"]], **kwargs)
    tail_length = kwargs.get("tail_length", 200)

    if tail_length > 300:
        layer._max_length = tail_length
        layer.tail_length = tail_length
    
    return layer


def sparsify_tracks(
    viewer: napari.Viewer,
    dense_tracks: np.ndarray,
    sparse_tracks: np.ndarray,
    *,
    start_time: int = 0,
    scale: np.ndarray,
    radius: float = 10.0,
    **kwargs,
) -> Tracks:

    dense_df = pd.DataFrame(dense_tracks, columns=["track_id", "t", "z", "y", "x"], copy=True)
    sparse_df = pd.DataFrame(sparse_tracks, columns=["track_id", "t", "z", "y", "x"], copy=True)

    dense_df = dense_df.query(f"t >= {start_time}")

    dense_df[["z", "y", "x"]] *= scale
    sparse_df[["z", "y", "x"]] *= scale

    selected_track_ids = []

    for t in range(int(sparse_df["t"].max() + 1)):

        sparse_slice = sparse_df.query(f"t == {t}")

        if len(sparse_slice) == 0:
            continue

        dense_slice = dense_df.query(f"t == {t}")

        dense_tree = KDTree(dense_slice[["z", "y", "x"]])
        dist, dense_indices = dense_tree.query(sparse_slice[["z", "y", "x"]], k=1)

        selected_track_ids.extend(dense_slice.iloc[dense_indices[dist < radius]]["track_id"])

    selected_track_ids = np.asarray(selected_track_ids)
    selected_track_ids = selected_track_ids.astype(int)
    selected_track_ids, counts = np.unique(selected_track_ids, return_counts=True)
    min_counts = 10
    selected_track_ids = selected_track_ids[counts > min_counts]

    tracks_df = dense_df[dense_df["track_id"].isin(selected_track_ids)]
    tracks_df[['z', 'y', 'x']] /= scale

    layer = viewer.add_tracks(tracks_df, scale=scale, **kwargs)

    tail_length = kwargs.get("tail_length", 200)

    if tail_length > 300:
        layer._max_length = tail_length
        layer.tail_length = tail_length
    
    return layer


def main() -> None:

    root = Path("<DEFINED BY USER>")
    T = 400

    viewer = napari.Viewer()
    resize_dask_cache(0)
    viewer.theme = "dark"

    kwargs = dict(visible=False, rendering="attenuated_mip", gamma=0.7)

    viewer.open(root / "stabilized.zarr", plugin="napari-ome-zarr", **kwargs)
    for l in viewer.layers[:2]:
        l.contrast_limits = (l.contrast_limits[0], 2000)
    
    # FULL RES
    for l in list(viewer.layers):
        data, state, type_str = l.as_layer_data_tuple()
        del state["multiscale"]
        viewer.layers.remove(l)
        viewer._add_layer_from_data(data[0], state, type_str)

    tracks_width = 3
    end_translation = np.asarray((0, 300, 1000))

    tracks_kwargs = dict(
        scale=viewer.layers[0].scale[-3:],
        colormap="hsv",
        tail_width=tracks_width,
        blending="translucent_no_depth",
        opacity=0.0,
    )

    load_tracks_layers(
        viewer,
        root / "deeplearning/tracking_ch0/results/tracks.csv",
        name="tracks_ch0_dl",
        tail_length=25,
        **tracks_kwargs,
    ).blending = "translucent"

    load_tracks_layers(
        viewer,
        root / "tracking_ch1/results/tracks.csv",
        name="tracks_ch1",
        tail_length=50,
        **tracks_kwargs,
    ).translate = end_translation

    layer = load_tracks_layers(
        viewer,
        root / "analysis/filtered_tracks/tracks_ch1_2024_06_25.csv",
        name="tracks_ch1_val",
        tail_length=1000,
        **tracks_kwargs,
    )
    layer.translate = end_translation

    layer = sparsify_tracks(
        viewer,
        viewer.layers["tracks_ch0_dl"].data,
        viewer.layers["tracks_ch1_val"].data,
        radius=10.0,
        name="tracks_ch0_dl_sparse",
        tail_length=1000,
        **tracks_kwargs,
    )
    # layer.blending = "additive"

    for l in viewer.layers:
        l._update_thumbnail = lambda *args, **kwargs: None

    viewer.window.resize(1800, 1000)

    viewer.dims.set_point(0, 0)
    viewer.dims.ndisplay = 3

    # viewer.camera.center = (415, 520, 845) # (175, 520, 590)
    viewer.camera.center = (466, 511, 903)
    viewer.camera.zoom = 0.9
    viewer.camera.angles = (20, 40, 120)

    for l in viewer.layers[:2]:
        l.visible = True

    step = 5
    viewer.layers["1"].translate = end_translation

    animation = Animation(viewer)

    animation.capture_keyframe()

    viewer.dims.set_point(0, 50)

    animation.capture_keyframe(50 * step)

    viewer.layers["tracks_ch0_dl"].opacity = 1.0
    viewer.layers["tracks_ch1"].opacity = 1.0
    viewer.dims.set_point(0, 75)

    animation.capture_keyframe((75 - 50) * step)

    viewer.dims.set_point(0, 150)

    animation.capture_keyframe((150 - 100) * step)

    viewer.layers["tracks_ch1"].opacity = 0
    viewer.layers["tracks_ch1_val"].opacity = 1.0
    viewer.dims.set_point(0, 175)

    animation.capture_keyframe((175 - 150) * step)

    viewer.dims.set_point(0, 250)

    animation.capture_keyframe((250 - 175) * step)

    viewer.layers["tracks_ch0_dl"].opacity = 0
    viewer.layers["tracks_ch0_dl_sparse"].opacity = 1.0
    viewer.dims.set_point(0, 275)

    animation.capture_keyframe((275 - 250) * step)

    for l in viewer.layers[:2]:
        l.opacity = 0.0

    viewer.dims.set_point(0, 400)

    animation.capture_keyframe((400 - 275) * step)

    animation.animate(FIG_DIR / "videos/sparse_embryo.mp4", fps=60)

    # viewer.close()
    # napari.run()


def example() -> None:

    root = Path("<DEFINED BY USER>")
    T = 400

    viewer = napari.Viewer()
    resize_dask_cache(0)
    viewer.theme = "dark"

    kwargs = dict(visible=False, rendering="attenuated_mip", gamma=0.7)

    viewer.open(root / "stabilized.zarr", plugin="napari-ome-zarr", **kwargs)
    for l in viewer.layers[:2]:
        l.contrast_limits = (l.contrast_limits[0], 2000)

    # viewer.open(root / "normalized.zarr", plugin="napari-ome-zarr", **kwargs)
    # for l in viewer.layers[:2]:
    #     l.contrast_limits = (0.1, 1)
    # viewer.layers[1].gamma = 0.5

    # viewer.open(root / "segmentation_ch0.zarr", plugin="napari-ome-zarr", **kwargs)

    # viewer.open(root / "segmentation_ch1.zarr", plugin="napari-ome-zarr", **kwargs)

    tracks_width = 2

    load_tracks_layers(
        viewer,
        root / "tracking_ch0/results/tracks.csv",
        scale=viewer.layers[0].scale,
        visible=False,
        colormap="hsv",
        blending="opaque",
        tail_length=100,
        tail_width=tracks_width,
        name="tracks_ch0",
    )

    load_tracks_layers(
        viewer,
        root / "deeplearning/tracking_ch0/results/tracks.csv",
        scale=viewer.layers[0].scale,
        visible=False,
        colormap="hsv",
        blending="opaque",
        tail_length=100,
        tail_width=tracks_width,
        name="tracks_ch0_dl",
    )

    load_tracks_layers(
        viewer,
        root / "tracking_ch1/results/tracks.csv",
        scale=viewer.layers[0].scale,
        visible=False,
        colormap="hsv",
        blending="opaque",
        tail_length=300,
        tail_width=tracks_width,
        name="tracks_ch1",
    )

    load_tracks_layers(
        viewer,
        root / "analysis/filtered_tracks/tracks_ch1_2024_05_03.csv",
        scale=viewer.layers[0].scale,
        visible=False,
        colormap="hsv",
        blending="opaque",
        tail_length=1000,
        tail_width=tracks_width,
        name="tracks_ch1_val",
    )

    sparsify_tracks(
        viewer,
        viewer.layers["tracks_ch0_dl"].data,
        viewer.layers["tracks_ch1_val"].data,
        viewer.layers[0].scale[-3:],
        radius=10.0,
        visible=False,
        colormap="hsv",
        blending="opaque",
        tail_length=1000,
        tail_width=tracks_width,
        name="tracks_ch0_dl_sparse",
    )

    for l in viewer.layers:
        l._update_thumbnail = lambda *args, **kwargs: None

    viewer.window.resize(1400, 1000)

    viewer.dims.set_point(0, T)

    viewer.dims.ndisplay = 3

    viewer.camera.center = (390, 415, 440)
    viewer.camera.zoom = 0.9
    viewer.camera.angles = (20, 40, 120)

    # viewer.close()
    napari.run()


if __name__ == "__main__":
    main()
