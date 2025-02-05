from pathlib import Path

import napari
import numpy as np
import pandas as pd
from skimage import io
from napari.layers import Labels
from ultrack.utils import labels_to_edges

import sys; sys.path.insert(0, "..");from constants import ULTRACK_DIR
from utils import add_scale_bar, contour_overlay


FIG_DIR = Path(".")
FIG_DIR.mkdir(exist_ok=True)


def invert(im: np.ndarray) -> np.ndarray:
    im = im - im.min()
    im = im / im.max()
    im = 1 - im
    return (im * 255).astype(np.uint8)


def crop_and_save(
    im: np.ndarray,
    bbox: tuple[int],
    im_path: Path,
    suffix: str,
) -> None:
    crop = im[bbox[0] : bbox[1], bbox[2] : bbox[3]]
    crop_path = im_path.parent / f"{im_path.stem}_{suffix}{im_path.suffix}"
    io.imsave(crop_path, crop)


def main() -> None:
    dataset = "01"
    data_dir = ULTRACK_DIR / "examples/parameter_sweeping"
    time = 25
    scale = (0.65, 0.65)
    print(data_dir)

    viewer = napari.Viewer()
    viewer.window.resize(1800, 1000)
    layer, = viewer.open(sorted((data_dir / f"Fluo-C2DL-Huh7/{dataset}").glob("*.tif")), stack=True)

    def load_and_save_labels(
        dir_name: Path,
        out_path: Path,
        image: np.ndarray,
        save_contour: bool,
    ) -> None:
        layer, = viewer.open(
            sorted((data_dir / dir_name / "TRA").glob("*.tif")),
            layer_type="labels",
            stack=True,
            name=out_path.name.removesuffix(".png"),
        )
        labels = layer.data
        labels_img = contour_overlay(image[time], labels[time], radius=5)
        io.imsave(out_path, labels_img)
        crop_and_save(labels_img, (428, 582, 870, 1024), out_path, "crop_1")
        crop_and_save(labels_img, (810, 964, 0, 154), out_path, "crop_2")

        det, cont = labels_to_edges(labels, sigma=5.0)

        if save_contour:
            io.imsave(FIG_DIR / f"detection_{out_path.name}", ~det[time])
            io.imsave(FIG_DIR / f"contour_{out_path.name}", invert(cont[time]))

    image = layer.data
    for gamma in [0.1, 0.25, 0.5, 1]:
        im = np.power(image, gamma)
        im = invert(im)
        io.imsave(FIG_DIR / f"image_{gamma}.png", im[time])
        load_and_save_labels(f"{dataset}_LABELS_{gamma}", FIG_DIR / f"labels_{gamma}.png", im, True)
    
    labels = []
    for layer in viewer.layers:
        if isinstance(layer, Labels):
            labels.append(layer.data)
        
    det, cont = labels_to_edges(labels, sigma=5.0)

    io.imsave(FIG_DIR / f"detection_combined.png", ~det[time])
    io.imsave(FIG_DIR / f"contour_combined.png", invert(cont[time]))
   
    load_and_save_labels(f"{dataset}_COMBINED", FIG_DIR / f"labels_combined.png", invert(image), False)
    for l in viewer.layers:
        l.visible = False
    
    # napari.run()

    viewer.layers["labels_combined"].visible = True
    viewer.layers["labels_combined"].data[:25, ...] = 0
    viewer.layers["labels_combined"].data[26:, ...] = 0
    viewer.layers["labels_combined"].scale = (10, 1, -1)
    viewer.layers["labels_combined"].bounding_box.line_color = "black"
    viewer.layers["labels_combined"].bounding_box.point_color = "black"
    viewer.layers["labels_combined"].bounding_box.visible = True

    viewer.open(
        data_dir / f"{dataset}_tracks.csv",
        plugin="ultrack",
        tail_width=2,
        tail_length=50,
        colormap="hsv",
        scale=(10, 1, -1),
        blending="opaque",
    )
    viewer.dims.ndisplay = 3
    # viewer.camera.angles = (0, 0, 180)
    # viewer.camera.angles = (115, 90, -65)
    viewer.camera.center = (90, 450, -475)
    viewer.camera.angles = (145, -45, -135)
    viewer.theme = "light"

    viewer.screenshot(FIG_DIR / f"tracks.png")

    scale_bar_path = FIG_DIR / f"reference_scale_img.png"
    io.imsave(scale_bar_path, np.full_like(image[time], 255, dtype=np.uint8))
    add_scale_bar(scale_bar_path, scale_bar_path, pixel_size=scale[0], bar_length=100, bar_color="black", label_color="black", font_size=24)

    df = pd.concat([pd.read_csv(data_dir / f"{ds}_scores.csv") for ds in ("01", "02")])
    df = df[["gamma", "TRA", "DET", "fp_nodes", "fn_nodes", "fp_edges", "fn_edges"]]
    print(df)
    print()
    print(df.groupby("gamma", dropna=False, as_index=False).mean())

    viewer.theme = "dark"
    viewer.close()


if __name__ == "__main__":
    main()
