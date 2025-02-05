from pathlib import Path

import pandas as pd
import numpy as np
import napari
import matplotlib.pyplot as plt
from napari_plot_profile._dock_widget import profile


FIG_DIR = Path(".")


def plot(x: np.ndarray, y: np.ndarray, outpath: Path, color: str = "white") -> None:
    # Create a new figure and axis
    fig, ax = plt.subplots()

    # # invert y axis
    # ax.invert_yaxis()

    # Plot the data with a white line
    ax.plot(x, y, color=color)

    # Set the title and labels with white color
    ax.set_title("Intensity profile", color=color)
    ax.set_xlabel("Line length", color=color)
    ax.set_ylabel("Image intensity", color=color)

    # Set the tick colors to white
    ax.tick_params(axis='x', colors=color)
    ax.tick_params(axis='y', colors=color)

    # Set the facecolor of the figure and axes to be transparent
    ax.set_facecolor('none')
    fig.set_facecolor('none')

    # Save the figure with a transparent background
    fig.savefig(outpath, transparent=True, dpi=300)


def main() -> None:
    viewer = napari.Viewer()

    layer, = viewer.open(
        "http://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS001.ome.zarr",
        plugin="napari-ome-zarr",
        rendering="attenuated_mip",
        gamma=0.7,
        colormap="magma",
    )
    viewer.window.resize(1800, 1000)

    image = layer.data[0][395, 221]
    viewer.layers.clear()

    im_layer = viewer.add_image(
        image,
        colormap="magma",
        rendering="attenuated_mip",
        gamma=0.7,
        contrast_limits=(25, 350),
        # scale=scale[-2:],
    )

    viewer.camera.center = (0.0, 1120, 2050)
    viewer.camera.zoom = 4

    viewer.screenshot(FIG_DIR / "slice_intensity.png")

    l1_layer = viewer.add_shapes(
        [[1028.78463349, 2056.97016013],
         [1078.86507626, 2102.94918733]],
         shape_type="line",
         edge_color="cyan",
         edge_width=2,
    )

    viewer.screenshot(FIG_DIR / "profile_1.png")

    prof1 = pd.DataFrame(profile(im_layer, l1_layer.data[0]))

    plot(prof1["distances"], prof1["intensities"], FIG_DIR / "profile_plot_1.png", "cyan")

    l2_layer = viewer.add_shapes(
        [[1115.49053458, 1982.49647672],
         [1163.58331918, 1988.74853872]],
         shape_type="line",
         edge_color="lime",
         edge_width=2,
    )

    prof2 = pd.DataFrame(profile(im_layer, l2_layer.data[0]))

    plot(prof2["distances"], prof2["intensities"], FIG_DIR / "profile_plot_2.png", "lime")

    viewer.screenshot(FIG_DIR / "profile_2.png")

    # napari.run()



if __name__ == "__main__":
    main()
