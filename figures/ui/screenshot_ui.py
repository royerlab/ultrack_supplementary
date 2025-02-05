import napari
import tempfile
import time
from ultrack.widgets import UltrackWidget
from ultrack import MainConfig
from rich import print


DATA_PATH = "<DEFINED BY USER>/multi-color-cytoplasm.tif"


def main() -> None:

    viewer = napari.Viewer()
    viewer.window.resize(1950, 900)
    viewer.theme = "light"

    widget = UltrackWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right")

    viewer.open(DATA_PATH, name="RGB video")
    viewer.camera.zoom = 0.55

    config = MainConfig()

    n_workers = 6

    config.data_config.working_dir = tempfile.mkdtemp()
    config.data_config.n_workers = n_workers
    config.segmentation_config.min_area = 500
    config.segmentation_config.n_workers = n_workers
    config.linking_config.n_workers = n_workers
    config.tracking_config.disappear_weight = -5
    print(config)

    # temporary working dir
    widget._data_forms.load_config(config)

    widget_config = widget._data_forms.get_config()  # trigger the update of the config
    print(widget_config)

    widget._bt_toggle_settings.setChecked(True)
    widget._bt_toggle_settings.clicked.emit()

    viewer.dims.set_point(0, 250)

    napari.run()

    widget._bt_run.clicked.emit()

    # time.sleep(1)
    # while widget._current_worker is not None and widget._current_worker.is_running:
    #     time.sleep(0.5)

    # # napari.run()
    # time.sleep(10)

    # viewer.screenshot("ui/ultrack_napari.png", canvas_only=False)

    viewer.theme = "dark"


if __name__ == "__main__":
    main()

