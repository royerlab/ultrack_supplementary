from pathlib import Path
import moviepy.editor
import moviepy.video
import moviepy.video.VideoClip
import pandas as pd
import moviepy
import math
import napari
from napari_animation import Animation
from napari.utils import resize_dask_cache
from moviepy.video.fx.freeze import freeze


from ultrack.tracks import get_subgraph, sort_track_ids, inv_tracks_df_forest
from ultrack.tracks.video import _disable_thumbnails, _block_refresh
from utils import remove_multiscale


def load_videos(dir_path: Path) -> dict[int, moviepy.editor.VideoFileClip]:
    """
    Load videos from a directory and return a dictionary with the video name and the track id.
    """
    video_dict = {}
    for video_path in dir_path.glob("*axis_0_roi.mp4"):
        if video_path.name.startswith("."):
            continue

        track_id = int(video_path.stem.split("_")[0])
        video_dict[track_id] = moviepy.editor.VideoFileClip(str(video_path))

    return video_dict


def compose_video(
    graph: dict[int, int],
    video_d: dict[int, moviepy.editor.VideoFileClip],
    sorted_ids: list[int],
) -> moviepy.editor.CompositeVideoClip:

    size = next(iter(video_d.values())).size
    print(size)

    n_ids = len(video_d)

    height = int(math.log2(n_ids + 1))
    n_leaves = n_ids // 2

    level = 0
    offset = 0
    step = 2
    by_level: dict[int, list[int]] = {i: [] for i in range(height)}

    while level < height:
        for l, j in enumerate(range(offset, n_ids, step)):
            track_id = sorted_ids[j]
            # pos = (j * size[0] / 2, (height - level - 1) * size[1])
            pos = (j * size[0] / 2, 0)
            video_d[track_id] = video_d[track_id].set_position(pos)
            by_level[level].append(track_id)

        offset += 2 ** level
        step = 2 ** (level + 2)
        level += 1
    
    total_length = 0
    node_id = sorted_ids[0]
    while node_id is not None:
        total_length += video_d[node_id].duration
        node_id = graph.get(node_id)

    for l in sorted(by_level.keys(), reverse=True):
        for j in by_level[l]:
            parent_id = graph.get(j)

            new_v = video_d[j]
            # new_v: moviepy.editor.VideoClip = new_v.fx(freeze, t="end", total_duration=total_length, padding_end=1)

            if parent_id is not None:
                parent_v = video_d[parent_id]
                # this function breaks the positions
                new_v = new_v.set_start(parent_v.end)
                new_v = new_v.set_position(video_d[j].pos)

            video_d[j] = new_v
            print(j, new_v.pos(0), new_v.start, new_v.end)
        print("----")
    
    # freezing last frames
    # for k, v in video_d.items():
    #     pos, start, end = v.pos, v.start, v.end
    #     v: moviepy.editor.VideoClip = v.fx(freeze, t="end", freeze_duration=total_length - end, padding_end=1)
    #     v = v.set_position(pos)
    #     v = v.set_start(start)
    #     video_d[k] = v
    
    cc = moviepy.editor.CompositeVideoClip(
        list(video_d.values()),
        # size=(int(size[0] * (n_leaves + 1)), int(size[1] * height)),
        size=(int(size[0] * (n_leaves + 1)), size[1]),
    )
    return cc


def embryo_thumbnail() -> None:
    viewer = napari.Viewer()
    viewer.window.resize(900, 400)
    resize_dask_cache(0)

    level = 3
    root = Path("<DEFINED BY USER>")

    out_path = Path("sparse_embryo/videos/comparison")
    out_path.mkdir(parents=True, exist_ok=True)

    viewer.open(
        root / "normalized.zarr",
        plugin="napari-ome-zarr",
        rendering="attenuated_mip",
        interpolation3d="nearest",
        attenuation=0.25,
        blending="additive",
    )
    scale = viewer.layers[0].scale[-3:]
    T = viewer.layers[0].data.shape[0]

    remove_multiscale(viewer, level=level)

    crops_dir = Path("<DEFINED BY USER>/analysis")
    tracks_df = pd.read_csv(crops_dir / "filtered_tracks/tracks_ch1_2024_06_25.csv")
    lineage_id = 13527
    tracks_df = get_subgraph(tracks_df, lineage_id)

    viewer.add_tracks(
        tracks_df[["track_id", "t", "z", "y", "x"]].to_numpy(),
        colormap="twilight_shifted",
        blending="opaque",
        name="tree",
        scale=scale,
        tail_width=10,
    )
    _disable_thumbnails(viewer)

    viewer.dims.ndisplay = 3
    step = 5

    for is_red_visibile in [True, False]:
        viewer.layers[1].visible = is_red_visibile
        viewer.dims.set_point(0, 0)
        animation = Animation(viewer)
        animation.capture_keyframe()

        viewer.dims.set_point(0, T)

        animation.capture_keyframe(step * T)

        animation.animate(
            filename=out_path / f"whole_{'both' if is_red_visibile else 'green'}.mp4",
            fps=60,
        )
    
    viewer.close()


def main() -> None:

    lineage_id = 13527

    crops_dir = Path("<DEFINED BY USER>/analysis")
    out_dir = Path(__file__).parent / "sparse_embryo/videos/comparison"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    both_ch_dir = crops_dir / f"gt_videos_both_2024_05_03_annot/{lineage_id}"
    green_ch_dir = crops_dir / f"gt_videos_green_2024_05_03_annot/{lineage_id}"

    # tracks_df = pd.read_csv(crops_dir / "filtered_tracks/tracks_ch1_2024_05_03.csv")
    tracks_df = pd.read_csv(crops_dir / "filtered_tracks/tracks_ch1_2024_06_25.csv")

    tracks_df = get_subgraph(tracks_df, lineage_id)
    sorted_ids = sort_track_ids(tracks_df)

    graph = inv_tracks_df_forest(tracks_df)

    both_v_d = load_videos(both_ch_dir)
    green_v_d = load_videos(green_ch_dir)

    both_v = compose_video(graph, both_v_d, sorted_ids)
    green_v = compose_video(graph, green_v_d, sorted_ids)

    both_v.write_videofile(str(out_dir / "both_ch.mp4"))
    green_v.write_videofile(str(out_dir / "green_ch.mp4"))

    # green_v = green_v.set_position((both_v.size[0], 0))
    green_v = green_v.set_position((0, both_v.size[1]))

    cc = moviepy.editor.CompositeVideoClip([both_v, green_v], size=(both_v.size[0], 2 * both_v.size[1]))

    cc.write_videofile(str(out_dir / "both_and_green_ch.mp4"))


if __name__ == "__main__":
    # embryo_thumbnail()
    main()

