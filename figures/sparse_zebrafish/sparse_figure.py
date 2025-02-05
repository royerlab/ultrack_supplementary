from pathlib import Path
import napari
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from napari.layers import Image, Tracks
from napari.utils import resize_dask_cache
from skimage.segmentation import relabel_sequential


FIG_DIR = Path(".")
FIG_DIR.mkdir(exist_ok=True)


def snapshot_tracks_at_slice(viewer: napari.Viewer, width: int = 5) -> None:

    layers = ["ch0", "ch0_dl", "ch1"]

    t, z = viewer.dims.current_step[:2]

    viewer.screenshot(FIG_DIR / f"closeup_t{int(t)}_z{int(z)}.png")

    for n in layers:

        df = pd.DataFrame(viewer.layers[n].data, columns=["track_id", "t", "z", "y", "x"])

        track_ids = df.query(f"t == {t} & z >= {z - width} & z <= {z + width}")["track_id"]

        viewer.add_tracks(
            df[df["track_id"].isin(track_ids)],
            colormap="hsv",
            blending="opaque",
            tail_length=200,
            tail_width=5,
            scale=viewer.layers[n].scale,
        )

        viewer.layers[f"segments_{n}"].visible = True

        viewer.screenshot(FIG_DIR / f"closeup_t{int(t)}_z{int(z)}_{n}.png")

        prev_visibility = {}

        for l in viewer.layers:
            if not isinstance(l, Tracks):
                prev_visibility[l.name] = l.visible
                l.visible = False

        viewer.screenshot(FIG_DIR / f"closeup_t{int(t)}_z{int(z)}_{n}_tracks.png")

        viewer.layers.remove_selected()

        for l, v in prev_visibility.items():
            viewer.layers[l].visible = v

        viewer.layers[f"segments_{n}"].visible = False


def closeup() -> None:

    root = Path("<DEFINED BY USER>")
    T = 400

    viewer = napari.Viewer()
    # viewer.scale_bar.visible = True
    viewer.window.resize(1400, 1000)
    resize_dask_cache(0)

    kwargs = {}

    viewer.open(root / "stabilized.zarr", plugin="napari-ome-zarr", **kwargs)

    viewer.open(
        root / "tracking_ch1/results/tracks.csv",
        plugin="ultrack",
        scale=viewer.layers[0].scale,
        visible=False,
        colormap="hsv",
        blending="opaque",
        tail_length=200,
        tail_width=5,
        name="ch1",
    )

    viewer.open(
        root / "tracking_ch0/results/tracks.csv",
        plugin="ultrack",
        scale=viewer.layers[0].scale,
        visible=False,
        colormap="hsv",
        blending="opaque",
        tail_length=200,
        tail_width=5,
        name="ch0",
    )

    viewer.open(
        root / "deeplearning/tracking_ch0/results/tracks.csv",
        plugin="ultrack",
        scale=viewer.layers[0].scale,
        visible=False,
        colormap="hsv",
        blending="opaque",
        tail_length=200,
        tail_width=5,
        name="ch0_dl",
    )

    viewer.open(
        root / "tracking_ch1/results/segments.zarr",
        scale=viewer.layers[0].scale,
        name="segments_ch1",
        visible=False,
    )[0].contour = 3

    viewer.open(
        root / "tracking_ch0/results/segments.zarr",
        scale=viewer.layers[0].scale,
        name="segments_ch0",
        visible=False,
    )[0].contour = 3

    viewer.open(
        root / "deeplearning/tracking_ch0/results/segments.zarr",
        scale=viewer.layers[0].scale,
        name="segments_ch0_dl",
        visible=False,
    )[0].contour = 3

    viewer.theme = "dark"
    viewer.camera.zoom = 10

    # viewer.dims.set_current_step((0, 1), (260, 252))
    # viewer.camera.center = (0.0, 730, 210)
    # snapshot_tracks_at_slice(viewer)

    viewer.layers["0"].contrast_limits = (400, 2500)

    viewer.dims.set_current_step((0, 1), (150, 406))
    viewer.camera.center = (0.0, 500, 275)
    snapshot_tracks_at_slice(viewer)

    viewer.layers["0"].contrast_limits = (400, 4500)

    viewer.dims.set_current_step((0, 1), (400, 419))
    viewer.camera.center = (0.0, 280, 390)
    snapshot_tracks_at_slice(viewer)

    viewer.dims.set_current_step((0, 1), (500, 61))
    viewer.camera.center = (0.0, 575, 460)
    snapshot_tracks_at_slice(viewer)

    viewer.close()


def load_tracks_layers(viewer: napari.Viewer, path: Path, **kwargs) -> None:
    df = pd.read_csv(path)
    df["track_id"], _, _ = relabel_sequential(df["track_id"].to_numpy(dtype=int))

    layer = viewer.add_tracks(df[["track_id", "t", "z", "y", "x"]], **kwargs)
    tail_length = kwargs.get("tail_length", 200)

    if tail_length > 300:
        layer._max_length = tail_length
        layer.tail_length = tail_length


def sparsify_tracks(
    viewer: napari.Viewer,
    dense_tracks: np.ndarray,
    sparse_tracks: np.ndarray,
    scale: np.ndarray,
    radius: float = 10.0,
    **kwargs,
) -> None:

    dense_df = pd.DataFrame(dense_tracks, columns=["track_id", "t", "z", "y", "x"], copy=True)
    sparse_df = pd.DataFrame(sparse_tracks, columns=["track_id", "t", "z", "y", "x"], copy=True)

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

    selected_track_ids = np.unique(selected_track_ids)

    tracks_df = dense_df[dense_df["track_id"].isin(selected_track_ids)]
    tracks_df[['z', 'y', 'x']] /= scale

    layer = viewer.add_tracks(tracks_df, scale=scale, **kwargs)

    tail_length = kwargs.get("tail_length", 200)

    if tail_length > 300:
        layer._max_length = tail_length
        layer.tail_length = tail_length


def embryo() -> None:

    root = Path("<DEFINED BY USER>")
    T = 400

    viewer = napari.Viewer()
    resize_dask_cache(0)
    viewer.theme = "dark"

    kwargs = dict(visible=False, rendering="attenuated_mip", gamma=0.7)

    # viewer.open(root / "stabilized.zarr", plugin="napari-ome-zarr", **kwargs)
    # for l in viewer.layers[:2]:
    #     l.contrast_limits = (l.contrast_limits[0], 2000)
    viewer.open(root / "normalized.zarr", plugin="napari-ome-zarr", **kwargs)
    for l in viewer.layers[:2]:
        l.contrast_limits = (0.1, 1)
    viewer.layers[1].gamma = 0.5

    viewer.open(root / "segmentation_ch0.zarr", plugin="napari-ome-zarr", **kwargs)

    viewer.open(root / "segmentation_ch1.zarr", plugin="napari-ome-zarr", **kwargs)

    load_tracks_layers(
        viewer,
        root / "tracking_ch0/results/tracks.csv",
        scale=viewer.layers[0].scale,
        visible=False,
        colormap="hsv",
        blending="opaque",
        tail_length=100,
        tail_width=5,
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
        tail_width=5,
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
        tail_width=5,
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
        tail_width=5,
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
        tail_width=5,
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

    for l in viewer.layers:
        if isinstance(l, Image):
            l.multiscale = False
            l.data = l.data[0]
    
    for l, n in zip(viewer.layers[:2], ("ch0", "ch1")):
        fig_path = FIG_DIR / f"embryo_{n}.png"
        if not fig_path.exists():
            l.visible = True
            viewer.screenshot(fig_path)
        l.visible = False

    fig_path = FIG_DIR / "embryo.png"
    if not fig_path.exists():
        for l in viewer.layers[:2]:
            l.visible = True
        viewer.screenshot(fig_path)

    for l in list(viewer.layers[:2]):
        l.visible = False
        viewer.layers.remove(l.name)
    
    viewer.layers[1].colormap = "gray_r"
    viewer.layers[1].rendering = "minip"
    viewer.layers[3].colormap = "gray_r"
    viewer.layers[3].rendering = "minip"
    
    for l, n in zip(list(viewer.layers[:4]), ("ch0_det", "ch0_edge", "ch1_det", "ch1_edge")):
        fig_path = FIG_DIR / f"embryo_{n}.png"
        if not fig_path.exists():
            l.visible = True
            viewer.screenshot(fig_path)
            l.visible = False
        viewer.layers.remove(l.name)

    for ch in ("ch0", "ch0_dl", "ch0_dl_sparse", "ch1", "ch1_val"):
        viewer.theme = "light"
        name = f"tracks_{ch}"
        viewer.layers[name].visible = True
        viewer.screenshot(FIG_DIR / f"embryo_{ch}_tracks.png")
        viewer.layers[name].visible = False

    viewer.theme = "dark"

    viewer.close()


if __name__ == "__main__":
    closeup()
    embryo()
