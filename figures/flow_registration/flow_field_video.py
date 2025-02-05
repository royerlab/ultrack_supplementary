from pathlib import Path

import zarr
import pandas as pd
import numpy as np
import napari
import matplotlib.pyplot as plt

from skimage import color
from skimage.filters import gaussian

import sys; sys.path.insert(0, "..");from constants import ULTRACK_DIR
from utils import simple_recording

from ultrack.imgproc.flow import advenct_from_quasi_random, trajectories_to_tracks
from ultrack.tracks.stats import tracks_df_movement


def create_intensity_bar(max_value: float, output_path: Path) -> None:
    # Create a gradient from 0 to max_value
    gradient = np.linspace(0, max_value, 256)[::-1].reshape(-1, 1)
    gradient = np.hstack((gradient, gradient))
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(2, 4))
    
    # Display the gradient
    ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap('magma'))
    
    # Remove x-axis ticks and label
    ax.set_axis_off()
    
    # Set title for the legend
    ax.text(1.2, -0.1, '0', transform=ax.transAxes, va='bottom', ha='right')
    ax.text(1.2, 1.1, str(max_value), transform=ax.transAxes, va='top', ha='right')
    
    # Save the figure as a PNG file
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5, dpi=300)
    # plt.show()


def create_hsv_circle(output_path: Path, diameter: float = 200) -> None:
    # Create a coordinate grid
    x, y = np.linspace(-1, 1, diameter), np.linspace(-1, 1, diameter)
    X, Y = np.meshgrid(x, y)

    # Convert cartesian to polar coordinates
    R, T = np.sqrt(X**2 + Y**2), np.arctan2(Y, X)
    T = (T + np.pi) / (2 * np.pi)  # Convert from [-pi, pi] to [0, 1]

    # Mask the values which are outside the circle
    mask = R <= 1

    # Create the HSV image
    H = T
    S = R
    V = np.ones_like(R)

    # Convert HSV image to RGB
    hsv_image = np.zeros((diameter, diameter, 3))
    hsv_image[..., 0] = H
    hsv_image[..., 1] = S
    hsv_image[..., 2] = V
    rgb_image = color.hsv2rgb(hsv_image)

    # Mask the values which are outside the circle
    rgb_image[~mask] = 1

    # Plot the image
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb_image)
    ax.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5, dpi=300)
    # plt.show()


def image_video() -> None:
    data_dir = ULTRACK_DIR / "examples/flow_field_3d"
    flow_field = zarr.open(data_dir / "flow.zarr")
    fig_dir = Path("flow_field")
    fig_dir.mkdir(exist_ok=True)

    # stack y, x with z-average
    y = - flow_field[:, -2].mean(axis=1) * flow_field.shape[-2]
    x = - flow_field[:, -1].mean(axis=1) * flow_field.shape[-1]

    # T, Y, X arrays
    y = gaussian(y, sigma=(0, 4, 4))
    x = gaussian(x, sigma=(0, 4, 4))

    angle = np.arctan2(y, x)
    length = np.sqrt(np.square(y) + np.square(x))

    create_intensity_bar(length.max(), fig_dir / "flow_length_legend.png")
    create_hsv_circle(fig_dir / "flow_angle_legend.png")

    viewer = napari.Viewer()
    viewer.window.resize(1800, 1000)

    viewer.add_image(length, colormap="magma", scale=(4, 4))
    # to record without image overlay
    # simple_recording(viewer, fig_dir / "flow_length.mp4")
    # viewer.layers.clear()

    layer, = viewer.open(
        sorted((data_dir / "Fluo-N3DL-TRIC/02").glob("*.tif")),
        stack=True,
        blending="additive",
    )
    layer.data = np.asarray(np.max(layer.data, axis=1))
    # simple_recording(viewer, fig_dir / "image.mp4")  # to record only the image
    simple_recording(viewer, fig_dir / "flow_length.mp4")

    # viewer.layers.clear()
    viewer.layers.remove(viewer.layers[0])

    layer = viewer.add_image(angle, colormap="hsv", scale=(4, 4), opacity=0.25, blending="opaque")
    viewer.layers.move(1, 0)
    simple_recording(viewer, fig_dir / "flow_angle.mp4")

    viewer.close()


def add_flow_tracks(
    viewer: napari.Viewer,
    flow_field: zarr.Array,
    shape: tuple[int, int, int],
    start: int,
    length: int,
    n_samples: int = 1_000,
    index: int = 0,
    **kwargs,
) -> pd.DataFrame:

    flow_field = flow_field[start : start + length]

    trajectory = advenct_from_quasi_random(flow_field, shape, n_samples=n_samples)
    flow_tracklets = pd.DataFrame(
        trajectories_to_tracks(trajectory),
        columns=["track_id", "t", "z", "y", "x"],
    )
    flow_tracklets["track_id"] += index * n_samples
    flow_tracklets["t"] += start
    flow_tracklets[["z", "y", "x"]] += 0.5  # napari was crashing otherwise, might be an openGL issue
    flow_tracklets[["dz", "dy", "dx"]] = tracks_df_movement(flow_tracklets)
    flow_tracklets["angles"] = np.arctan2(flow_tracklets["dy"], flow_tracklets["dx"])

    viewer.add_tracks(
        flow_tracklets[["track_id", "t", "z", "y", "x"]],
        name="flow vectors",
        visible=True,
        tail_length=25,
        features=flow_tracklets[["angles", "dy", "dx"]],
        colormap="hsv",
        **kwargs,
    ).color_by="angles"


def flow_tracks_video() -> None:
    data_dir = ULTRACK_DIR / "examples/flow_field_3d"
    flow_field = zarr.open(data_dir / "flow.zarr")
    fig_dir = Path("flow_field")
    fig_dir.mkdir(exist_ok=True)

    viewer = napari.Viewer()
    viewer.window.resize(1800, 1000)

    length = 50
    for i, t in enumerate(range(0, 100, length // 2)):
        add_flow_tracks(viewer, flow_field, flow_field.shape, t, length, index=i)

    simple_recording(
        viewer,
        fig_dir / "flow_tracks.mp4",
        t_length=flow_field.shape[0] - 1,
    )
    viewer.close()


if __name__ == "__main__":
    flow_tracks_video()
    image_video()
