from pathlib import Path
import numpy as np
import napari


def main() -> None:
    fig_dir = Path(".")

    data_dir = Path("<DEFINED BY USER>")

    runs = [
        ("01_RES", ""),
        ("07_RES", "_phase_cp")
    ]

    for ds, suffix in runs:
        viewer = napari.Viewer()
        viewer.window.resize(1800, 800)
        viewer.theme = "light"

        viewer.open(data_dir / ds, plugin="napari-ctc-io")
        lb_l = viewer.layers["labels"]
        data = lb_l.data
        new_data = np.zeros((data.shape[0], *data.shape), dtype=data.dtype)
        for t in range(data.shape[0]):
            new_data[t, t] = data[t]
        lb_l.data = new_data
        lb_l.rendering = "translucent"

        tracks_l = viewer.layers["tracks"]
        tracks_l.blending = "opaque"

        lb_l.bounding_box.line_color = "black"
        lb_l.bounding_box.point_color = "black"
        lb_l.bounding_box.visible = True

        t_scale = 10
        
        for l in viewer.layers:
            l.scale = (t_scale, 1, 1)

        viewer.dims.set_point(0, lb_l.data.shape[0])
        viewer.layers.move(0, -1)

        viewer.dims.ndisplay = 3

        viewer.camera.center = (230, 995, 1067)
        viewer.camera.zoom = 0.4
        viewer.camera.angles = (25, -46, -33)

        viewer.screenshot(fig_dir / f"tracks_3D{suffix}.png")

        # napari.run()

        viewer.theme = "dark"
        viewer.close()


if __name__ == "__main__":
    main()
