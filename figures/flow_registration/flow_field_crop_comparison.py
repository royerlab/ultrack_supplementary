from pathlib import Path

import dask.array as da
import napari
import numpy as np
import zarr
#from flow_field_video import add_flow_tracks
import sys; sys.path.insert(0, "..");from constants import ULTRACK_DIR

def main() -> None:
    data_dir = ULTRACK_DIR / "examples/flow_field_3d"
    res_dir = Path(
        "<DEFINED BY USER>"
    )
    fig_dir = Path("flow_field")
    fig_dir.mkdir(exist_ok=True)
    step = 5
    time = 147

    viewer = napari.Viewer()
    viewer.theme = "dark"
    viewer.window.resize(1600, 800)
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

    viewer.layers["image"].data = np.array(viewer.layers["image"].data)

    for mode in ["flow", "no_flow"]:

        #labels = zarr.open(f"no_flow_segments_{step}.zarr")
        labels = zarr.open(res_dir / f"{mode}_segments_{step}.zarr")

        labels = da.from_array(labels)
        # labels = np.asarray(labels)
        if mode == "flow":
            labels = np.where(np.logical_or(labels == 3716, labels == 7741), labels, 0)
        elif mode == "no_flow":
            labels = np.where(np.logical_or(labels == 5263, labels == 2567), labels, 0)

        viewer.add_labels(labels, name=f"{mode}_segments_{step}", visible=True, scale=(step, 1, 1, 1)).contour = 5

        viewer.dims.set_current_step(range(2), (time, 4))

        viewer.open(
             res_dir / f"{mode}_tracks_{step}.csv",
            #f"{mode}_tracks_{step}.csv",
            plugin="ultrack",
            tail_length=3,
            tail_width=15,
            name=f"tracks_{mode}",
            scale=(step, 1, 1, 1),
        )

        # viewer.camera.center = (0.0, 2100, 820)
        viewer.camera.center = (4.0, 876, 905)
        viewer.camera.zoom = 10

        viewer.screenshot(fig_dir / f"{mode}_crop_first.png")
        #
        viewer.dims.set_current_step(0, time + step)
        viewer.screenshot(fig_dir / f"{mode}_crop_second.png")

        viewer.dims.set_current_step(0, time + 2 * step)
        viewer.screenshot(fig_dir / f"{mode}_crop_third.png")

        viewer.layers[f"tracks_{mode}"].visible = False
        viewer.layers[f"{mode}_segments_{step}"].visible = False

        # viewer.camera.zoom = 4.0
        # viewer.camera.center = (0, 1700, 1450)
        # # viewer.screenshot(fig_dir / "tracks_crop.png")

        # viewer.layers["tracks"].graph = {}
        # viewer.layers["tracks"].data = viewer.layers["tracks"].data

        # viewer.reset_view()
        # viewer.screenshot(fig_dir / "tracks_frame.png")

        #viewer.theme = "dark"

    # napari.run()
    viewer.close()


if __name__ == "__main__":
    main()
