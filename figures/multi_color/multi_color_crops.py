
import napari
import zarr
import dask.array as da

from multi_color_constants import ROOT_DIR, FIG_DIR, CTC_RES_DIR


def main() -> None:
    img = da.from_zarr(zarr.open(ROOT_DIR / "normalized.zarr"))
    scale = (0.75469,) * 2

    viewer = napari.Viewer()
    viewer.window.resize(1600, 1000)

    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"

    viewer.add_image(img, rgb=True, scale=scale, visible=False)
    viewer.add_image(img, channel_axis=-1, scale=scale, colormap="gray", name="gray_img", visible=True)

    for p in ROOT_DIR.glob("*_labels.zarr"):
        lbs = da.from_zarr(zarr.open(p))
        name = p.stem.removesuffix("_labels")
        if lbs.ndim == 4:
            for i in range(lbs.shape[-1]):
                viewer.add_labels(lbs[..., i], name=f"{name}_{i}", scale=scale, visible=False).contour = 3
        else:
            viewer.add_labels(lbs, name=name, scale=scale, visible=False).contour = 3

    viewer.open(ROOT_DIR / "curated_tracks/segments.tif", scale=scale, name="combined_labels", visible=False, layer_type="labels")[0].contour = 3
    
    # gray watershed vs gray cellpose
    viewer.camera.zoom = 8 
    viewer.camera.center = (500, 825)

    fig_dir = FIG_DIR / "comparison"
    fig_dir.mkdir(exist_ok=True)

    viewer.dims.set_point(0, 261)

    viewer.layers["gray_pure_ws"].new_colormap(1)
    viewer.layers["gray_pure_ws"].visible = True
    viewer.screenshot(fig_dir / "gray_ws_vs_cp.png")
    viewer.layers["gray_pure_ws"].visible = False

    viewer.layers["gray_cellpose"].new_colormap(42)
    viewer.layers["gray_cellpose"].visible = True
    viewer.screenshot(fig_dir / "gray_cp_vs_ws.png")
    viewer.layers["gray_cellpose"].visible = False

    viewer.dims.set_point(0, 261)
    viewer.camera.center = (522, 360)

    # viewer.layers["gray_pure_ws"].new_colormap(123)
    viewer.layers["gray_pure_ws"].visible = True
    viewer.screenshot(fig_dir / "ws_gray_vs_color.png")
    viewer.layers["gray_pure_ws"].visible = False

    viewer.dims.set_point(0, 280)
    viewer.camera.center = (150, 1050)

    viewer.layers["gray_cellpose"].visible = True
    viewer.screenshot(fig_dir / "cp_gray_vs_color.png")
    viewer.layers["gray_cellpose"].visible = False

    for l in viewer.layers:
        if l.name.startswith("gray_img"):
            l.visible = False
    
    cp_layers = ["cellpose_0", "cellpose_1", "cellpose_2"]
    ws_layers = ["ws_pure_0", "ws_pure_1", "ws_pure_2"]

    viewer.layers["img"].visible = True

    viewer.dims.set_point(0, 261)
    viewer.camera.center = (522, 360)

    for l in ws_layers:
        viewer.layers[l].visible = True
    viewer.screenshot(fig_dir / "ws_color_vs_gray.png")
    for l in ws_layers:
        viewer.layers[l].visible = False

    viewer.dims.set_point(0, 280)
    viewer.camera.center = (150, 1050)

    for l in cp_layers:
        viewer.layers[l].visible = True
    viewer.screenshot(fig_dir / "cp_color_vs_gray.png")
    for l in cp_layers:
        viewer.layers[l].visible = False

    viewer.dims.set_point(0, 286)
    viewer.camera.center = (630, 350)

    for l in ws_layers:
        viewer.layers[l].visible = True
    viewer.screenshot(fig_dir / "ws_vs_ultrack.png")
    for l in ws_layers:
        viewer.layers[l].visible = False

    for l in cp_layers:
        viewer.layers[l].visible = True
    viewer.screenshot(fig_dir / "cp_vs_ultrack.png")
    for l in cp_layers:
        viewer.layers[l].visible = False
    
    viewer.layers["combined_labels"].visible = True
    viewer.screenshot(fig_dir / "ultrack_vs_cp_vs_ws.png")
    viewer.layers["combined_labels"].visible = False

    for l in list(viewer.layers):
        if l.name != "img":
            viewer.layers.remove(l)
     
    for p in CTC_RES_DIR.iterdir():
        if p.stem.startswith("COMBINED"):
            ls = viewer.open(p, plugin="napari-open-ctc", scale=scale, name=p.name, visible=False)
            ls[0].contour = 3
            ls[1].colormap = "twilight_shifted"
            ls[1].tail_width = 15
            ls[1].tail_length = 100
    
    viewer.camera.zoom = 10
    viewer.dims.set_point(0, 236)
    viewer.camera.center = (490, 350)

    for l in viewer.layers:
        if "NORMAL_LINKS" in l.name:
            l.visible = True

    viewer.screenshot(fig_dir / "track_new_cell.png")

    viewer.dims.set_point(0, 238)
    viewer.screenshot(fig_dir / "track_new_cell_normal_link.png")

    for l in viewer.layers:
        if "NORMAL_LINKS" in l.name:
            l.visible = False
        else:
            l.visible = True
    viewer.screenshot(fig_dir / "track_new_cell_color_link.png")

    # napari.run()
    viewer.close()


if __name__ == "__main__":
    main()
