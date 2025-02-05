from pathlib import Path
from typing import Optional, Tuple

import napari
import numpy as np
from numpy.typing import ArrayLike
from napari_animation import Animation
from PIL import Image, ImageDraw, ImageFont
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, disk


def simple_recording(
    viewer: napari.Viewer,
    output_path: Path,
    capture_factor: int = 5,
    t_length: Optional[int] = None,
) -> None:

    viewer.dims.set_point(0, 0)

    animation = Animation(viewer)
    animation.capture_keyframe()

    if t_length is None:
        t_length = viewer.layers[0].data.shape[0] - 1

    viewer.dims.set_point(0, t_length)

    animation.capture_keyframe(t_length * capture_factor)

    animation.animate(output_path, fps=60)


def remove_multiscale(viewer: napari.Viewer, level: int = 0) -> None:
    layers = list(viewer.layers)
    for l in layers:
        if hasattr(l, "multiscale") and l.multiscale:
            data, kwargs, type_name = l.as_layer_data_tuple()
            data = data[level]
            kwargs["scale"] = [s * 2 ** level for s in kwargs["scale"][-3:]]
            del kwargs["multiscale"]
            viewer.layers.remove(l)
            viewer._add_layer_from_data(data, kwargs, type_name)


def add_scale_bar(
    image_path: str, 
    output_path: str, 
    pixel_size: float, 
    bar_length: float, 
    bar_height: int = 5, 
    bar_color: str = 'white', 
    label_color: str = 'white', 
    position: Tuple[int, int] = (-50, -50),
    font_size: int = 15,
) -> None:
    """
    Add a scale bar to the given image.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    output_path : str
        Path to save the image with scale bar.
    pixel_size : float
        Size of one pixel in units (e.g., micrometers).
    bar_length : float
        Length of the scale bar in the same units as pixel_size.
    bar_height : int, optional
        Height of the scale bar in pixels. Default is 5 pixels.
    bar_color : str, optional
        Color of the scale bar. Default is 'white'.
    label_color : str, optional
        Color of the label text. Default is 'white'.
    position : tuple of int, optional
        Tuple (x, y) indicating the top-left position of the scale bar. Default is (10, 10).

    Examples
    --------
    >>> add_scale_bar("input.jpg", "output_with_scale.jpg", 0.5, 100)
    """

    # Calculate the length of the scale bar in pixels
    bar_length_pixels = int(bar_length / pixel_size)
    
    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    if position[0] < 0:
        position = (image.width + position[0] - bar_length_pixels, position[1])

    if position[1] < 0:
        position = (position[0], image.height + position[1])

    # Draw the scale bar
    bar_top_left = position
    bar_bottom_right = (position[0] + bar_length_pixels, position[1] + bar_height)
    draw.rectangle([bar_top_left, bar_bottom_right], fill=bar_color)

    # Draw the scale bar label (optional, but good to have)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    label_position = (bar_top_left[0], bar_top_left[1] - 20)
    draw.text(label_position, f"{bar_length} units", fill=label_color, font=font)
    
    # Save the image
    image.save(output_path)


def rgb_to_cmy(image: ArrayLike, channel_axis: int = -1) -> ArrayLike:
    """
    Convert an RGB image to a Cyan, Yellow, Magenta (CMY) image.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image.
    channel_axis : int, optional
        Axis representing the color channels in the image. Default is -1 (last axis).

    Returns
    -------
    np.ndarray
        Converted CMY image.
    """
    
    # Ensure the image is of type uint8 with values between 0 and 255
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    if channel_axis != -1:
        image = np.moveaxis(image, channel_axis, -1)
    
    # Convert RGB to CMY
    cmy_image = 255 - image

    if channel_axis != -1:
        cmy_image = np.moveaxis(cmy_image, -1, channel_axis)

    return cmy_image


def contour_overlay(
    image: ArrayLike,
    labels: ArrayLike,
    radius: int = 3,
) -> ArrayLike:
    image = np.asarray(image)
    labels = np.asarray(labels)
    border = find_boundaries(labels, connectivity=1, mode="inner")
    border = binary_dilation(border, disk(radius))
    border[labels <= 0] = False
    labels[~border] = 0

    labeled_img = label2rgb(
        labels,
        image=image / 255,
        bg_label=0,
        alpha=0.5,
    )
    labeled_img = (labeled_img * 255).astype(np.uint8)
    labeled_img[~border] = image[~border, None]
    return labeled_img