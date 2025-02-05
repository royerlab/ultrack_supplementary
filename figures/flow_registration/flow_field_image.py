from pathlib import Path

import napari
import zarr
from flow_field_video import add_flow_tracks
import sys; sys.path.insert(0, "..");from constants import ULTRACK_DIR


def main() -> None:
    mode = "flow"
    data_dir = ULTRACK_DIR / "examples/flow_field_3d"
    res_dir = Path(
        "<DEFINED BY USER>"
    )
    fig_dir = Path("flow_field")
    fig_dir.mkdir(exist_ok=True)

    viewer = napari.Viewer()
    viewer.theme = "dark"
    viewer.window.resize(1600, 1000)
    viewer.dims.ndisplay = 3
    # viewer.theme = "light"

    viewer.open(
        sorted((data_dir / "Fluo-N3DL-TRIC/02").glob("*.tif")),
        stack=True,
        contrast_limits=(300, 3_500),
        # gamma=0.5,
        gamma=1.0,
        name="image",
    )

    labels = zarr.open(res_dir / "segments.zarr")

    viewer.add_labels(labels, name="segments", visible=False)

    viewer.dims.set_point(range(2), (157, 8))

    viewer.dims.set_point(0, 158)

    prev_layer = viewer.add_image(
        viewer.layers["image"].data[157],
        colormap="green",
        blending="additive",
        contrast_limits=(300, 3_500),
        gamma=1.0,
    )
    viewer.layers["image"].colormap = "red"

    viewer.reset_view()
    viewer.screenshot(fig_dir / "combined_image.png")

    viewer.layers.remove(prev_layer)
    viewer.layers["image"].colormap = "gray"

    flow_field = zarr.open(res_dir / f"{mode}_1.zarr")
    # using an extra time point before
    add_flow_tracks(viewer, flow_field, labels.shape[-3:], 156, 2, n_samples=5_000, tail_width=3)

    viewer.reset_view()
    viewer.screenshot(fig_dir / "flow_frame.png")

    for layer in list(viewer.layers):
        if layer.name not in ("image", "segments"):
            viewer.layers.remove(layer)
    viewer.theme = "dark"

    viewer.close()
    # napari.run()


if __name__ == "__main__":
    main()