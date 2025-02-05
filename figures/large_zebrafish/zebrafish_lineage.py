from pathlib import Path
from typing import Optional

import pandas as pd
import napari
import zarr
from ultrack.tracks import get_subgraph
from napari_animation import Animation

FIG_DIR = Path(".")


def make_track_video(
    viewer: napari.Viewer,
    ds: zarr.Group,
    tree: pd.DataFrame,
    output_dir: Path,
    channel: str,
    z_scale: float,
) -> None:

    track_id = int(tree["track_id"].iloc[0])
    output_dir = output_dir / str(track_id)
    output_dir.mkdir(exist_ok=True)

    time_factor = 5
    img_kwargs = dict(gamma=0.7, contrast_limits=(0, 500), colormap="magma")
    tracks_kwargs = dict(colormap="twilight", tail_length=300)

    for proj_axis in (0, 1):
        out_path = output_dir / f"proj_{proj_axis}.mp4"

        if out_path.exists():
            print(f"Skipping {out_path}, it already exists.")
            continue

        array = ds[f"{channel}_projection_{proj_axis}"]

        if proj_axis == 1:
            proj_scale = (-z_scale, 1)
            proj_tree = tree[["track_id", "t", "z", "x"]].to_numpy()
        else:
            proj_scale = (1, 1)
            proj_tree = tree[["track_id", "t", "y", "x"]].to_numpy()

        # viewer = napari.Viewer()
        viewer.window.resize(1814, 1203)

        viewer.add_image(array, **img_kwargs, scale=proj_scale)
        viewer.add_tracks(proj_tree, **tracks_kwargs, scale=proj_scale, tail_width=5)
        viewer.reset_view()

        viewer.dims.set_point(1, 0) # setting to projection slice

        t_min, t_max = tree["t"].min(), tree["t"].max()

        viewer.dims.set_point(0, t_min)

        animation = Animation(viewer)
        animation.capture_keyframe()

        viewer.dims.set_point(0, t_max)

        animation.capture_keyframe((t_max - t_min) * time_factor)
        animation.animate(out_path, fps=60)
        # viewer.close()
        viewer.layers.clear()

    array = ds[channel]
    scale = None # (z_scale, 1, 1)

    # viewer = napari.Viewer()
    viewer.window.resize(1814, 1203)

    viewer.add_image(array, **img_kwargs, scale=scale)
    # viewer.add_labels(segm).contour = 2
    viewer.reset_view()

    viewer.camera.zoom = 8.0
    step = 10

    for group_id, track_group in tree.groupby("track_id", sort=True):
        out_path = output_dir / f"{group_id}_roi.mp4"
        if out_path.exists():
            print(f"Skipping {out_path}, it already exists.")
            continue

        viewer.add_tracks(
            track_group[["track_id", "t", "z", "y", "x"]].to_numpy(),
            **tracks_kwargs,
            tail_width=2,
            scale=scale,
            name="tracks",
        )

        animation = Animation(viewer)

        for i, (_, group) in enumerate(track_group.groupby("t", sort=True, as_index=False)):
            if i % step != 0:
                continue

            pos = group[["t", "z", "y", "x"]].mean().to_numpy()
            viewer.dims.set_point(range(4), pos)
            viewer.camera.center = pos[-3:]
            animation.capture_keyframe(step * time_factor)

        pos = group[["t", "z", "y", "x"]].mean().to_numpy()
        viewer.dims.set_point(range(4), pos)
        viewer.camera.center = pos[-3:]
        animation.capture_keyframe((i % step) * time_factor)

        animation.animate(out_path, fps=60)

        viewer.layers.remove("tracks")

    viewer.layers.clear()


def load_lineage(
    root_dir: Path,
    scale: Optional[tuple[float]] = None,
) -> tuple[pd.DataFrame, dict[int, int]]:

    df = pd.read_csv(root_dir / "distributed_tracking/results/filtered_tracks.csv")
    if scale is not None:
        df[["z", "y", "x"]] *= scale

    df["track_id"] = df["track_id"].astype(int)

    # tree = get_subgraph(df, 416558)  # very close
    bad_tracklets = (8189, 8193, 8190, 8176, 8167, 8168, 8169, 8172)
    tree = get_subgraph(df, 8154)

    graph = {
        8194: 8154,
        8158: 8154,
        8170: 8158,
        8171: 8158,
    }

    tree = tree[(tree["track_id"] != 8171) | (tree["t"] >= 767)]

    tree = tree[~tree["track_id"].isin(bad_tracklets)]
    tree = tree[tree["t"] < 788]

    tree.loc[tree["track_id"] == 8157, "track_id"] = 8158
    tree.loc[tree["track_id"] == 8159, "track_id"] = 8158
    tree.loc[tree["track_id"] == 8164, "track_id"] = 8158
                                       
    tree.loc[tree["track_id"] == 8173, "track_id"] = 8171

    return tree, graph


def make_crop_movie() -> None:
    root_dir = Path("<DEFINED BY USER>")

    group = zarr.open(root_dir / "segmentation/stabilized_m2.zarr")["fused"]
    print(group)
    
    tree, _ = load_lineage(root_dir)

    viewer = napari.Viewer()
    viewer.window.resize(1800, 1000)

    # t 
    # first div.
    # T = 209 & 212
    # sec div.
    # T = 765 & 768

    make_track_video(viewer, group, tree, FIG_DIR, "fused", 3)

    viewer.close()


def screenshot_lineage() -> None:
    root_dir = Path("<DEFINED BY USER>")

    scale = (1.24, 0.439, 0.439)
    image = zarr.open(root_dir / "segmentation/stabilized_m2.zarr")["fused"]["fused"]
    
    tree, graph = load_lineage(root_dir, scale)

    viewer = napari.Viewer()

    viewer.add_image(image, scale=scale, gamma=0.7, colormap="gray", contrast_limits=(25, 250))
    viewer.window.resize(1600, 1000)

    viewer.dims.set_point(0, 750)
    viewer.dims.ndisplay = 3

    viewer.camera.zoom = 1.1
    viewer.camera.angles = (180, 10, -120)
    viewer.camera.perspective = 30.0

    viewer.add_tracks(
        tree[["track_id", "t", "z", "y", "x"]],
        tail_length=400,
        tail_width=50,
        blending="opaque",
        colormap="hsv",
        graph=graph,
    )
    # t 
    # first div.
    # T = 209 & 212
    # sec div.
    # T = 765 & 768

    viewer.camera.center = (290, 475, 480)
    tps = (150, 209, 212, 765, 768)

    for t in tps:
        if t > 500:
            viewer.camera.zoom = 1.2
            viewer.camera.center = (265, 605, 590)
            viewer.layers["image"].contrast_limits = (25, 450)

        viewer.dims.set_point(0, t)
        viewer.screenshot(FIG_DIR / f"lineage_3d_t{t}.png")

    viewer.dims.ndisplay = 2
    viewer.camera.zoom = 25

    viewer.layers[-1].visible = False
    
    for t in tps:
        viewer.dims.set_point(0, t)
        crop_df = tree[tree["t"] == t]
        for group_id, group in crop_df.groupby("track_id", sort=True):
            viewer.dims.set_point(1, group["z"].mean())
            viewer.camera.center = (0, group["y"].mean(), group["x"].mean())
            l = viewer.add_points(group[["t", "z", "y", "x"]], size=15, face_color="transparent", edge_color="red", n_dimensional=True)
            viewer.screenshot(FIG_DIR / f"lineage_2d_t{t}_track{group_id}.png")
            viewer.layers.remove(l)

    # napari.run(); return
    viewer.close()


def inspect_lineage() -> None:
    root_dir = Path("<DEFINED BY USER>")

    scale = (1.24, 0.439, 0.439)
    image = zarr.open(root_dir / "segmentation/stabilized_m2.zarr")["fused"]["fused"]
    
    tree, graph = load_lineage(root_dir, scale)

    viewer = napari.Viewer()

    viewer.add_image(image, scale=scale, gamma=0.7, colormap="magma", contrast_limits=(25, 450))
    viewer.window.resize(1800, 1000)

    viewer.dims.set_point(0, 750)
    viewer.scale_bar.visible = True

    viewer.camera.center = (0, 715, 325)
    viewer.camera.zoom = 3.5

    viewer.add_tracks(
        tree[["track_id", "t", "z", "y", "x"]],
        tail_length=400,
        blending="opaque",
        colormap="hsv",
        graph=graph,
    )

    viewer.add_points(
        tree[["t", "z", "y", "x"]],
        size=6,
        face_color="transparent",
        edge_color="white",
        n_dimensional=True,
    )

    # (t=329, z=406)
    # camera.center = (0.0, 626.7507829742981, 789.726713219712)
    # camera.zoom = 12.5

    napari.run(); return
    viewer.close()


if __name__ == "__main__":
    # inspect_lineage()
    make_crop_movie()
    # screenshot_lineage()
