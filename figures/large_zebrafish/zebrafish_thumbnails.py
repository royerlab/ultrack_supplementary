from pathlib import Path

import pandas as pd
import napari
import zarr
from napari.layers import Image, Tracks


FIG_DIR = Path(".")
COLORMAP = "inferno"


def segmentation_slice() -> None:
    segm = zarr.open(
        "<DEFINED BY USER>/distributed_tracking/results/segments.zarr"
    )

    segm_dir = Path("<DEFINED BY USER>/segmentation")

    image = zarr.open(segm_dir / "segmentation.zarr")["fused"]["fused"]
    contour = zarr.open(segm_dir / "blur.pred.zarr")["Boundary"]["Boundary"]
    prediction = zarr.open(segm_dir / "blur.pred.zarr")["Prediction"]["Prediction"]

    scale = (1.24, 0.439, 0.439)
    t = 750
    z_slice = 223

    viewer = napari.Viewer()

    im_layer = viewer.add_image(image, scale=scale, gamma=0.7, colormap=COLORMAP)
    viewer.add_image(contour, scale=scale)
    viewer.add_image(prediction, scale=scale)
    viewer.add_labels(segm, scale=scale, name="segms")

    df = pd.read_csv(
        "http://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS001_tracks.csv"
    )

    df[["z", "y", "x"]] /= scale

    pad = 7

    # Define the bounding box ranges
    z_min, z_max = z_slice - pad, z_slice  + pad
    # Filter the DataFrame based on the bounding box
    filtered_df = df[
        ((df['t'] - t).abs() < 1) &
        (df['z'] >= z_min) & (df['z'] < z_max)
    ]

    filtered_df = df[df["TrackID"].isin(filtered_df["TrackID"].unique())]

    viewer.add_tracks(
        filtered_df[["TrackID", "t", "z", "y", "x"]].to_numpy(),
        tail_length=50,
        blending="opaque",
        colormap="hsv",
        visible=False,
        name="tracks",
        scale=scale,
    )

    viewer.window.resize(1800, 1000)

    viewer.dims.set_current_step(1, z_slice)

    viewer.dims.set_point(0, t)
    viewer.scale_bar.visible = True

    for l in viewer.layers:
        if isinstance(l, Image):
            l.reset_contrast_limits()
        l.visible = False

    viewer.layers["image"].constrast_limits = (25, 250)

    viewer.camera.center = (0, 710, 325)
    viewer.camera.zoom = 3.5

    for l in viewer.layers:
        if not isinstance(l, Tracks):
            l.visible = True
            viewer.screenshot(FIG_DIR / f"slice_zebrafish_{l.name}.png")
            l.visible = False

    # tracks
    im_layer.colormap = "gray"
    im_layer.visible = True
    viewer.layers["tracks"].visible = True
    viewer.screenshot(FIG_DIR / f"slice_zebrafish_tracks.png")
    viewer.layers["tracks"].visible = False
    im_layer.visible = False
    im_layer.colormap = "inferno"

    viewer.window.resize(1500, 1000)

    viewer.camera.center = (0, 760, 390)
    viewer.camera.zoom = 30

    for l in viewer.layers:
        if not isinstance(l, Tracks):
            l.visible = True
            viewer.screenshot(FIG_DIR / f"slice_zebrafish_{l.name}_zoom.png")
            l.visible = False

    # tracks
    im_layer.colormap = "gray"
    im_layer.visible = True
    viewer.layers["tracks"].visible = True
    viewer.screenshot(FIG_DIR / f"slice_zebrafish_tracks_zoom.png")

    # napari.run()

    viewer.close()


def whole_embryo() -> None:
    MULTISCALE = False

    segm = zarr.open(
        "<DEFINED BY USER>/distributed_tracking/results/segments.zarr"
    )

    segm_dir = Path("<DEFINED BY USER>/segmentation")
    contour = zarr.open(segm_dir / "blur.pred.zarr")["Boundary"]["Boundary"]
    prediction = zarr.open(segm_dir / "blur.pred.zarr")["Prediction"]["Prediction"]

    viewer = napari.Viewer()

    im_layer, = viewer.open(
        "http://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS001.ome.zarr",
        plugin="napari-ome-zarr",
        rendering="attenuated_mip",
        gamma=0.7,
        colormap=COLORMAP,
        visible=False,
    )
    im_layer.contrast_limits = (25, 250)

    viewer.window.resize(1800, 1000)

    # disabling multiscale
    if not MULTISCALE:
        for layer in viewer.layers:
            layer.multiscale = False
            layer.data = layer.data[0]

    df = pd.read_csv("http://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS001_tracks.csv")

    viewer.add_tracks(
        df[["TrackID", "t", "z", "y", "x"]].to_numpy(),
        tail_length=50,
        blending="opaque",
        visible=False,
        name="tracks",
    )

    viewer.dims.set_point(0, 750)
    viewer.dims.ndisplay = 3

    viewer.camera.center = (265, 605, 590)
    viewer.camera.zoom = 1.2
    viewer.camera.angles = (180, 10, -120)
    viewer.camera.perspective = 30.0

    viewer.add_labels(segm, scale=viewer.layers[0].scale, name="segms", blending="opaque", rendering="translucent", visible=False)
    viewer.add_image(contour, scale=viewer.layers[0].scale, visible=False, rendering="mip")
    viewer.add_image(prediction, scale=viewer.layers[0].scale, visible=False, rendering="attenuated_mip", attenuation=1)

    im_layer.visible = True
    viewer.screenshot(FIG_DIR / "whole_zebrafish.png")

    im_layer.colormap = "gray"
    viewer.layers["tracks"].visible = True

    viewer.screenshot(FIG_DIR / "whole_zebrafish_tracks.png")

    for l in viewer.layers[-3:]:
        l.visible = True
        viewer.screenshot(FIG_DIR / f"whole_zebrafish_{l.name}.png")
        l.visible = False

    viewer.close()


def pipeline() -> None:
    MULTISCALE = True

    segm_dir = Path("<DEFINED BY USER>/segmentation")

    image = zarr.open(segm_dir / "stabilized_m2.zarr")["fused"]["fused"]
    contour = zarr.open(segm_dir / "blur.pred.zarr")["Boundary"]["Boundary"]
    prediction = zarr.open(segm_dir / "blur.pred.zarr")["Prediction"]["Prediction"]

    segm = zarr.open(segm_dir.parent / "distributed_tracking/results/segments.zarr")

    image = image[650, 200:230, 1710:1810, 1640:1740]
    contour = contour[650, 200:230, 1710:1810, 1640:1740]
    prediction = prediction[650, 200:230, 1710:1810, 1640:1740]
    segm = segm[650, 200:230, 1710:1810, 1640:1740]

    scale = (1.24, 0.439, 0.439)

    viewer = napari.Viewer()

    viewer.add_image(
        image,
        rendering="attenuated_mip",
        gamma=0.7,
        colormap=COLORMAP,
        contrast_limits=(25, 250),
        scale=scale,
        attenuation=0.25,
    )
    viewer.add_image(contour, visible=False, scale=scale, rendering="attenuated_mip", attenuation=1, gamma=2.0)
    viewer.add_image(prediction, visible=False, scale=scale, rendering="attenuated_mip", attenuation=1)
    viewer.add_labels(segm, scale=scale, visible=False, blending="minimum", rendering="translucent")

    df = pd.read_csv(
        "http://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS001_tracks.csv"
    )

    df[["z", "y", "x"]] /= scale

    pad = 1
    # Define the bounding box ranges
    z_min, z_max = 200  - pad, 230  + pad
    y_min, y_max = 1710 - pad, 1810 + pad
    x_min, x_max = 1640 - pad, 1740 + pad

    # Filter the DataFrame based on the bounding box
    filtered_df = df[
        ((df['t'] - 650).abs() < 1) &
        (df['z'] >= z_min) & (df['z'] < z_max) &
        (df['y'] >= y_min) & (df['y'] < y_max) &
        (df['x'] >= x_min) & (df['x'] < x_max)
    ]

    filtered_df = df[df["TrackID"].isin(filtered_df["TrackID"].unique())]

    filtered_df[['z', 'y', 'x']] -= (z_min, y_min, x_min)
    filtered_df[['z', 'y', 'x']] *= scale

    viewer.add_tracks(
        filtered_df[["TrackID", "t", "z", "y", "x"]].to_numpy(),
        translate=(filtered_df["t"].min(), 0, 0, 0),
        tail_length=50,
        blending="opaque",
        colormap="hsv",
        visible=False,
        name="tracks",
    )

    viewer.window.resize(1600, 1000)

    viewer.dims.ndisplay = 3

    # viewer.camera.center = (20, 25, 20)
    viewer.camera.center = (19.5, 22.75, 22)
    viewer.camera.zoom = 12
    viewer.camera.angles = (-25, 40, 150)
    viewer.camera.perspective = 30.0

    for name in ("image", "contour", "prediction", "segm"):
        viewer.layers[name].visible = True
        viewer.screenshot(FIG_DIR / f"3d_crop_zebrafish_{name}.png")
        viewer.layers[name].visible = False

    viewer.layers["tracks"].visible = True
    viewer.layers["image"].visible = True
    viewer.layers["image"].colormap = "gray"
    viewer.dims.set_current_step(0, 438)  # I don't know why this is not 650

    viewer.screenshot(FIG_DIR / f"3d_crop_zebrafish_tracks.png")
    
    # napari.run()
    viewer.close()


if __name__ == "__main__":
    whole_embryo()
    segmentation_slice()
    pipeline()
