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
    step = 5
    time = 157

    viewer = napari.Viewer()
    viewer.theme = "dark"
    viewer.window.resize(1600, 1000)
    # viewer.dims.ndisplay = 3
    # viewer.theme = "light"

    viewer.open(
        sorted((data_dir / "Fluo-N3DL-TRIC/02").glob("*.tif")),
        stack=True,
        contrast_limits=(300, 1_700),
        # gamma=0.5,
        gamma=1.0,
        name="image",
    )

    labels = zarr.open(res_dir / f"{mode}_segments_{step}.zarr")

    viewer.add_labels(labels, name=f"segments_{step}", visible=True, scale=(step, 1, 1, 1)).contour = 2

    viewer.dims.set_current_step(range(2), (time, 4))

    viewer.open(
        # data_dir / f"{mode}_tracks_{step}.csv",
        res_dir / f"{mode}_tracks_{step}.csv",
        plugin="ultrack",
        tail_length=2,
        name="tracks",
        scale=(step, 1, 1, 1),
    )

    # viewer.camera.center = (0.0, 2100, 820)
    viewer.camera.center = (0.0, 700, 930)
    viewer.camera.zoom = 3.5

    viewer.screenshot(fig_dir / "crop_first.png")

    viewer.dims.set_current_step(0, time + step)
    viewer.screenshot(fig_dir / "crop_second.png")

    viewer.dims.set_current_step(0, time + 2 * step)
    viewer.screenshot(fig_dir / "crop_third.png")

    # viewer.layers["tracks"].visible = True

    # viewer.camera.zoom = 4.0
    # viewer.camera.center = (0, 1700, 1450)
    # # viewer.screenshot(fig_dir / "tracks_crop.png")

    # viewer.layers["tracks"].graph = {}
    # viewer.layers["tracks"].data = viewer.layers["tracks"].data

    # viewer.reset_view()
    # viewer.screenshot(fig_dir / "tracks_frame.png")

    viewer.theme = "dark"

    napari.run()
    viewer.close()


if __name__ == "__main__":
    main()