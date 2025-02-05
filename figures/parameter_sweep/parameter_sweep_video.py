from pathlib import Path
from contextlib import nullcontext

import napari
import numpy as np
import pandas as pd
import skimage.morphology as morph
from ultrack.utils import labels_to_edges
from ultrack.tracks.video import (
    _disable_thumbnails,
    _optimize_layers,
)
from napari_animation import Animation


from tifffile import imread, imwrite


FIG_DIR = Path(".")
FIG_DIR.mkdir(exist_ok=True)


def generate_data(data_dir: Path, dataset: str) -> None:

    disk = morph.disk(3)
    for gamma in [0.1, 0.25, 0.5, 1]:
        cp_data = imread(data_dir / f"{dataset}_cellpose_{gamma}.tif")
        foreground, contour = labels_to_edges(cp_data)
        contour = morph.binary_dilation(contour, disk[None, ...])
        imwrite(data_dir / f"{dataset}_contour_{gamma}.tif", contour)
        imwrite(data_dir / f"{dataset}_foreground_{gamma}.tif", foreground)


def main() -> None:
    dataset = "01"
    data_dir =  Path("<DEFINED BY USER>")
    time = 25
    scale = (0.65, 0.65)
    print(data_dir)

    viewer = napari.Viewer()
    viewer.window.resize(1800, 1000)

    layer, = viewer.open(
        sorted((data_dir / f"Fluo-C2DL-Huh7/{dataset}").glob("*.tif")),
        stack=True,
        colormap="magma"
    )

    # generate_data(data_dir, dataset)
    translate = np.zeros(3)

    kwargs = dict(blending="additive", translate=translate, opacity=0.0)

    # for gamma in [0.1, 0.25, 0.5, 1]:
    for gamma in [1, 0.5, 0.25, 0.1, "combined"]:

        if not isinstance(gamma, str):
            cp_data = imread(data_dir / f"{dataset}_cellpose_{gamma}.tif")[:, None]
            viewer.add_labels(
                cp_data,
                name=f"cellpose {gamma}",
                opacity=0.0,
                translate=translate,
                rendering="translucent",
            )

        foreground = imread(data_dir / f"{dataset}_foreground_{gamma}.tif")[:, None]
        contour = imread(data_dir / f"{dataset}_contour_{gamma}.tif")[:, None]

        viewer.add_image(foreground, name=f"foreground {gamma}", **kwargs)
        viewer.add_image(contour, name=f"contour {gamma}", **kwargs)

        viewer.add_labels(
            imread(data_dir / f"{dataset}_labels_{gamma}.tif")[:, None],
            name=f"labels {gamma}",
            visible=False,
            translate=translate,
            rendering="translucent",
        )

        tracks_layer, = viewer.open(
            data_dir / f"{dataset}_tracks_{gamma}.csv",
            plugin="ultrack",
            colormap="hsv",
            blending="opaque",
            visible=False,
            translate=translate,
            name=f"ultrack {gamma}",
        )
        df = pd.DataFrame(tracks_layer.data, columns=["track_id", "t", "y", "x"])
        df["z"] = df["t"] * -15
        tracks_layer.data = df[["track_id", "t", "z", "y", "x"]].to_numpy()

        data, im_kwargs, layer_type = layer.as_layer_data_tuple()

        if gamma != "combined":

            data = np.power(np.asarray(data, dtype=np.float32), gamma)
            data -= data.min()
            data /= data.max()

            for k, v in kwargs.items():
                im_kwargs[k] = v
            im_kwargs["opacity"] = 1.0
            im_kwargs["contrast_limits"] = (0, 1) # (0.05 ** gamma, 1)
            im_kwargs["name"] = f"image {gamma}"

            viewer._add_layer_from_data(data[:, None], im_kwargs, layer_type)

            translate[-1] += data.shape[-1]

    viewer.layers.remove(layer)

    _disable_thumbnails(viewer)

    with nullcontext():
        _optimize_layers(viewer)

        viewer.layers["contour combined"].translate = translate / 2 + (1000, 0, 0)
        viewer.layers["foreground combined"].translate = translate / 2 + (-1000, 0, 0)
        viewer.layers["ultrack combined"].translate = translate / 2
        viewer.layers["labels combined"].translate = translate / 2

        animation = Animation(viewer)

        viewer.dims.set_point(0, 0)

        viewer.dims.ndisplay = 3
        viewer.camera.center = (0, 460, 2000)
        viewer.camera.zoom = 0.32
        viewer.camera.angles = (0.0, 0.0, 90)
        animation.capture_keyframe()

        viewer.dims.set_point(0, data.shape[0])

        animation.capture_keyframe(180)

        for l in viewer.layers:
            if "cellpose" in l.name:
                l.opacity = 1.0

        animation.capture_keyframe(120)

        animation.capture_keyframe(60)

        viewer.camera.center = (91, 482, 2323)
        viewer.camera.zoom = 0.27
        viewer.camera.angles = (30, -45, 145)

        animation.capture_keyframe(60)

        for l in viewer.layers:
            if "contour" in l.name and "combined" not in l.name:
                l.translate = np.asarray(l.translate) + (0, 1000, 0, 0)
                l.opacity = 1.0

            elif "foreground" in l.name and "combined" not in l.name:
                l.translate = np.asarray(l.translate) + (0, -1000, 0, 0)
                l.opacity = 1.0

        animation.capture_keyframe(80)

        animation.capture_keyframe(10)

        for l in viewer.layers:
            if "contour" in l.name and "all" not in l.name:
                l.translate = translate / 2 + (1000, 0, 0)

            if "foreground" in l.name and "all" not in l.name:
                l.translate = translate / 2 + (-1000, 0, 0)

        viewer.layers["contour combined"].opacity = 1.0
        viewer.layers["foreground combined"].opacity = 1.0

        animation.capture_keyframe(120)

        for l in viewer.layers:
            if "contour" in l.name and "combined" not in l.name:
                l.opacity = 0.0

            if "foreground" in l.name and "combined" not in l.name:
                l.opacity = 0.0

        animation.capture_keyframe(10)

        for l in viewer.layers:
            if "cellpose" in l.name or "image" in l.name:
                if l.name != "image 1":
                    l.opacity = 0.0

        animation.capture_keyframe(60)
        
        viewer.layers["image 1"].translate = translate / 2

        animation.capture_keyframe(30)

        for l in viewer.layers:
            if "combined" not in l.name and l.name != "image 1":
                l.visible = False
 
        viewer.layers["contour combined"].translate = translate / 2
        viewer.layers["foreground combined"].translate = translate / 2

        viewer.dims.set_point(0, 0)

        viewer.camera.zoom = 0.6
        viewer.camera.center = (-280, 550, 2600)
        # viewer.camera.center = (530, 780, 2315)

        animation.capture_keyframe()

        viewer.layers["contour combined"].visible = False
        viewer.layers["foreground combined"].visible = False
        viewer.layers["ultrack combined"].visible = True
        viewer.layers["labels combined"].visible = True

        viewer.layers["image 1"].translate = translate / 2 + (df["z"].min(), 0, 0)
        viewer.layers["labels combined"].translate = translate / 2 + (df["z"].min(), 0, 0)

        viewer.dims.set_point(0, data.shape[0])
    
        animation.capture_keyframe(120)

        animation.capture_keyframe(10)

    video_path = FIG_DIR / "method_example.mp4"

    animation.animate(video_path, fps=60)

    # napari.run()

    return


if __name__ == "__main__":
    main()
