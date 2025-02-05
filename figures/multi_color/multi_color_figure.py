from pathlib import Path

import napari
import zarr
import numpy as np
from tifffile import imread
from skimage import io
from napari.utils.color import ColorValue
from napari.layers import Tracks
import skimage.morphology as morph
from skimage.color import label2rgb

from ultrack.utils import labels_to_edges
from multi_color_constants import FIG_DIR

import sys
sys.path.insert(0, "..")

from constants import ULTRACK_DIR
from utils import add_scale_bar


def to_rgb(viewer: napari.Viewer) -> None:
    assert len(viewer.layers) == 3
    viewer.layers[0].colormap = "red"
    viewer.layers[1].colormap = "green"
    viewer.layers[2].colormap = "blue"

def old_main() -> None:
    time = 299
    img_path = "<DEFINE BY USER>"
    data_dir = ULTRACK_DIR / "examples/multi_color_ensemble"
    print(data_dir)

    img = imread(img_path)[time][None, ...]
    scale = (0.75469,) * 2

    viewer = napari.Viewer()
    viewer.window.resize(1800, 1000)

    viewer.scale_bar.unit = "um"
    viewer.scale_bar.color = "white"
    viewer.scale_bar.visible = True

    viewer.add_image(img, channel_axis=-1, gamma=0.75, scale=scale)

    norm = np.power(img[0].copy(), 0.75)
    norm = norm - norm.min()
    norm = norm / norm.max()
    # rgb to cmy
    # frame = np.zeros_like(norm)
    # for i in range(3):
    #     channel = [255] * 3
    #     channel[i] = 0
    #     frame += norm[..., i, None] * channel
    # frame[..., [1, 2]] = frame[..., [2, 1]]  # cmy to cym

    # already in rgb
    frame = norm  * 255

    frame = np.clip(frame, 0, 255)
    frame = frame.astype(np.uint8)
    im_path = FIG_DIR / "image.png"
    io.imsave(im_path, frame)
    add_scale_bar(
        im_path,
        im_path,
        pixel_size=0.75469,
        bar_length=100,
    ) 

    # viewer.screenshot(fig_dir / "image.png")
    viewer.scale_bar.visible = False

    # IMAGE
    step = 400
    for i, l in enumerate(viewer.layers):
        l.translate = (i * step, 0, 0)
        l.blending = "translucent"

    viewer.theme = "light"
    viewer.dims.ndisplay = 3
    viewer.camera.angles = (-10, 40, 155)
    viewer.camera.zoom = 0.6
    viewer.screenshot(FIG_DIR / "image_stacked.png")

    # NORMALIZED
    normalized = zarr.open(data_dir / "normalized.zarr")
    viewer.add_image(
        normalized[time][None, ...],
        channel_axis=-1,
        scale=scale,
        blending="opaque",
    )
    for l in viewer.layers[:3]:
        viewer.layers.remove(l.name)

    to_rgb(viewer)

    for i, l in enumerate(viewer.layers):
        l.translate = (i * step, 0, 0)

    viewer.screenshot(FIG_DIR / "normalized_stack.png")

    # WS
    ws_labels = zarr.open(data_dir / "ws_labels.zarr")
    layers = viewer.add_image(
        ws_labels[time][None, ...] > 0,
        scale=scale,
        channel_axis=-1,
        blending="translucent",
    )
    for c, layer in enumerate(layers):
        layer.translate = (c * step, 0, 0)
        layer.bounding_box.line_color = "black"  # ColorValue("black")
        layer.bounding_box.visible = True
        layer.bounding_box.points = False    
    
    for l in viewer.layers[:3]:
        viewer.layers.remove(l.name)

    to_rgb(viewer)

    viewer.screenshot(FIG_DIR / "ws_labels_stacked.png")

    # CELLPOSE
    cellpose_labels = zarr.open(data_dir / "cellpose_labels.zarr")
    layers = viewer.add_image(
        cellpose_labels[time][None, ...] > 0,
        scale=scale,
        channel_axis=-1,
        blending="translucent",
    )
    for c, layer in enumerate(layers):
        layer.translate=(c * step, 0, 0)
        layer.bounding_box.line_color = "black"  # ColorValue("black")
        layer.bounding_box.visible = True
        layer.bounding_box.points = False

   
    for l in viewer.layers[:3]:
        viewer.layers.remove(l.name)    

    to_rgb(viewer)

    viewer.screenshot(FIG_DIR / "cellpose_labels_stacked.png")

    all_labels = [cellpose_labels[time, ..., c][None, None, ...] for c in range(3)] +\
        [ws_labels[time, ..., c][None, None, ...] for c in range(3)]

    #saving UCM
    det, contour = labels_to_edges(all_labels)
    det = ((1 - det[0, 0]) * 255).astype(np.uint8)
    contour = contour[0, 0] / np.max(contour)
    contour = morph.dilation(contour, footprint=morph.disk(2))
    contour = ((1 - contour) * 255).astype(np.uint8)

    io.imsave(FIG_DIR / "detection.png", det)
    io.imsave(FIG_DIR / "contour.png", contour)

    # TRACKS
    viewer.layers.clear()

    viewer.dims.ndisplay = 2

    viewer.add_image(
        normalized,
        channel_axis=-1,
        scale=scale,
    )
    to_rgb(viewer)

    res_path = Path(img_path[:-4])

    viewer.open(
        res_path / "segments.tif",
        scale=scale,
        layer_type="labels",
    )[0].contour = 3

    viewer.open(
        res_path / "tracks.csv",
        plugin="ultrack",
        scale=scale,
        tail_width=3,
        tail_length=50,
        blending="opaque",
        colormap="hsv",
    )

    viewer.dims.set_point(0, time)

    viewer.screenshot(FIG_DIR / "tracks.png")

    viewer.theme = "dark"

    viewer.scale_bar.visible = True
    viewer.scale_bar.colored = True

    viewer.camera.zoom = 3.5
    viewer.camera.center = (0.0, 485, 1250)

    viewer.screenshot(FIG_DIR / "tracks_zoom.png")

    # napari.run()

    viewer.close()


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    rgb = label2rgb(labels, bg_label=0, bg_color=(1, 1, 1))
    return (255 * rgb).astype(np.uint8)


def _process_contours(det: np.ndarray, contour: np.ndarray) -> np.ndarray:
    det = ((1 - det[0, 0]) * 255).astype(np.uint8)
    contour = contour[0, 0] / np.max(contour)
    contour = morph.dilation(contour, footprint=morph.disk(2))
    contour = ((1 - contour) * 255).astype(np.uint8)
    return det, contour


def main() -> None:
    time = 299
    img_path = "<DEFINE BY USER>"
    data_dir = ULTRACK_DIR / "examples/multi_color_ensemble"
    print(data_dir)

    img = zarr.open(data_dir / "normalized.zarr")
    scale = (0.75469,) * 2

    viewer = napari.Viewer()
    viewer.window.resize(1800, 1000)

    norm = img[time]
    norm = norm - norm.min()
    norm = norm / norm.max()

    # already in rgb
    frame = norm  * 255

    frame = np.clip(frame, 0, 255)
    frame = frame.astype(np.uint8)
    im_path = FIG_DIR / "image.png"
    io.imsave(im_path, frame)
    add_scale_bar(
        im_path,
        im_path,
        pixel_size=0.75469,
        bar_length=100,
    ) 

    for c, color in enumerate(np.eye(3)):
        colored = (frame[..., c, None] * color[None, None, :]).astype(np.uint8)
        io.imsave(FIG_DIR / f"channel_{c}.png", colored )

    # WS
    ws_labels = zarr.open(data_dir / "ws_labels.zarr")

    # CELLPOSE
    cellpose_labels = zarr.open(data_dir / "cellpose_labels.zarr")

    for c in range(3):
        io.imsave(FIG_DIR / f"ws_{c}.png", labels_to_rgb(ws_labels[time, ..., c]))
        io.imsave(FIG_DIR / f"cellpose_{c}.png", labels_to_rgb(cellpose_labels[time, ..., c]))

        det, contour = labels_to_edges(ws_labels[time, ..., c][None, None, ...])
        det, contour = _process_contours(det, contour)
        io.imsave(FIG_DIR / f"ws_{c}_contour.png", contour)
        io.imsave(FIG_DIR / f"ws_{c}_detection.png", det)

        det, contour = labels_to_edges(cellpose_labels[time, ..., c][None, None, ...])
        det, contour = _process_contours(det, contour)
        io.imsave(FIG_DIR / f"cellpose_{c}_contour.png", contour)
        io.imsave(FIG_DIR / f"cellpose_{c}_detection.png", det)
            
    all_labels = [cellpose_labels[time, ..., c][None, None, ...] for c in range(3)] +\
        [ws_labels[time, ..., c][None, None, ...] for c in range(3)]
    
    #saving UCM
    det, contour = labels_to_edges(all_labels)
    det, contour = _process_contours(det, contour)

    io.imsave(FIG_DIR / "detection.png", det)
    io.imsave(FIG_DIR / "contour.png", contour)

    # TRACKS
    viewer.layers.clear()

    viewer.dims.ndisplay = 2

    viewer.add_image(
        img,
        channel_axis=-1,
        scale=scale,
    )
    to_rgb(viewer)

    res_path = Path(img_path[:-4])

    viewer.open(
        res_path / "segments.tif",
        scale=scale,
        layer_type="labels",
    )[0].contour = 3

    viewer.open(
        res_path / "tracks.csv",
        plugin="ultrack",
        scale=scale,
        tail_width=3,
        tail_length=50,
        blending="opaque",
        colormap="hsv",
    )

    viewer.dims.set_point(0, time)
    viewer.reset_view()

    viewer.camera.zoom = 0.95

    viewer.screenshot(FIG_DIR / "tracks.png")

    viewer.theme = "dark"

    viewer.scale_bar.visible = True
    viewer.scale_bar.colored = True

    viewer.camera.zoom = 3.5
    viewer.camera.center = (0.0, 485, 1250)

    viewer.screenshot(FIG_DIR / "tracks_zoom.png")

    for l in viewer.layers:
        if not isinstance(l, Tracks):
            l.data = l.data[time]
            l.visible = False
        else:
            l.bounding_box.line_color = "black"
            l.bounding_box.point_color = "black"
            l.bounding_box.visible = True
    
    viewer.dims.ndisplay = 3
    viewer.camera.zoom = 0.6
    viewer.camera.angles = (-165, 45, 20)
    viewer.theme = "light"

    viewer.layers[-2].rendering = "translucent"
    viewer.layers[-2].visible = True
    viewer.layers[-2].contour = 0
    viewer.layers.move(-2, -1)

    viewer.screenshot(FIG_DIR / "tracks_3d.png")

    viewer.theme = "dark"
    viewer.close()


if __name__ == "__main__":
    main()
    # old_main()
