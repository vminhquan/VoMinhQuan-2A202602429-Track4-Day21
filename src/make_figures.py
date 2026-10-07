"""Vẽ biểu đồ và ảnh minh hoạ cho REPORT (Topic A). Chạy sau `src.calib_drift`.

    python -m src.make_figures
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calib_drift import edge_distance_map, edge_score
from starter.datasets import load_frame
from starter.projection import draw_box2d, overlay_points, perturb_extrinsic, project_velo_to_image

FIG = Path("results/figures")
KITTI, NUSC = "data/kitti_mini", "data/nuscenes_mini_subset"


def render(fr, calib, label_filter=None, radius=2):
    uv, depth, mask = project_velo_to_image(fr["points"], calib, fr["image"].shape)
    vis = overlay_points(fr["image"], uv, depth, radius=radius)
    for obj in fr["labels"]:
        if obj.type != "DontCare" and (label_filter is None or label_filter(obj)):
            vis = draw_box2d(vis, obj.bbox)
    return vis, uv, depth


def put(img, text, y=28, scale=0.8):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
    return img


def crop(img, bbox, pad=40, size=(320, 240)):
    x1, y1, x2, y2 = bbox.astype(int)
    H, W = img.shape[:2]
    c = img[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]
    return cv2.resize(c, size, interpolation=cv2.INTER_NEAREST)


def fig_distance_demo():
    """Basic: 3 object ở 3 khoảng cách (frame 000049), cột = baseline / yaw 1° / yaw 2°."""
    fr = load_frame(KITTI, "000049")
    picks = {"near 5.5 m (Pedestrian)": 5.5, "mid 22.3 m (Car)": 22.3, "far 35.7 m (Car)": 35.7}
    objs = {k: min(fr["labels"], key=lambda o: abs(np.linalg.norm(o.location[[0, 2]]) - r))
            for k, r in picks.items()}
    rows = []
    for name, obj in objs.items():
        cells = []
        for yaw in (0.0, 1.0, 2.0):
            vis, _, _ = render(fr, perturb_extrinsic(fr["calib"], yaw_deg=yaw), lambda o: o is obj)
            cells.append(put(crop(vis, obj.bbox), f"{name} | yaw {yaw:g} deg", y=20, scale=0.45))
        rows.append(np.hstack(cells))
    cv2.imwrite(str(FIG / "demo_distance_yaw_000049.png"), np.vstack(rows))
    vis, _, _ = render(fr, fr["calib"])
    cv2.imwrite(str(FIG / "demo_overlay_000049.png"), put(vis, "KITTI 000049 baseline calib"))


def fig_plots():
    """Good/Advanced: in_box theo mức perturb (tách theo khoảng cách) + edge_ratio, cho 2 dataset."""
    for ds in ("kitti", "nuscenes"):
        s = pd.read_csv(f"results/drift_{ds}_summary.csv")
        s["factor"] = s.config.str.split("_").str[0]
        s["level"] = s.config.str.extract(r"_([\d.]+)")[0].astype(float)
        base = s[s.config == "baseline"].iloc[0]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, col, title in [(axes[0], "in_box_far_pct", "% điểm object rơi đúng 2D box — far (>30 m)"),
                               (axes[1], "in_box_near_pct", "% điểm object rơi đúng 2D box — near (<15 m)"),
                               (axes[2], "edge_ratio_mean", "edge_ratio (1 = như calib gốc)")]:
            for fac in ("yaw", "pitch", "roll"):
                d = s[s.factor == fac].sort_values("level")
                ax.plot([0, *d.level], [base[col], *d[col]], marker="o", label=fac)
            ax.set_xlabel("độ lệch (deg)")
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=.3)
        axes[2].axhline(0.9, color="k", ls="--", lw=1, label="ngưỡng 0.9")
        axes[0].legend(); axes[2].legend()
        fig.suptitle(f"{ds}: tham số perturb_extrinsic (roll quanh x, pitch quanh y, yaw quanh z của LiDAR)")
        fig.tight_layout()
        fig.savefig(FIG / f"drift_plot_{ds}.png", dpi=110)
        plt.close(fig)


def fig_fail_edge_miss():
    """Failure 1: yaw 1° ở KITTI 000015, edge score gần như không đổi nhưng object mid lệch khỏi box."""
    fr = load_frame(KITTI, "000015")
    dm = edge_distance_map(fr["image"])
    peds = [o for o in fr["labels"] if o.type == "Pedestrian" and np.linalg.norm(o.location[[0, 2]]) > 20]
    x1, y1 = np.min([o.bbox[:2] for o in peds], axis=0)
    x2, y2 = np.max([o.bbox[2:] for o in peds], axis=0)
    group = np.array([x1, y1, x2, y2])
    panels, zooms = [], []
    for yaw in (0.0, 1.0):
        vis, uv, depth = render(fr, perturb_extrinsic(fr["calib"], yaw_deg=yaw))
        score, _ = edge_score(uv, depth, dm)
        panels.append(put(vis, f"KITTI 000015 yaw {yaw:g} deg   edge_score={score:.3f}"))
        zooms.append(put(crop(vis, group, pad=25, size=(fr["image"].shape[1] // 2, 300)), f"3 Pedestrian ~24 m, yaw {yaw:g} deg", scale=0.6))
    cv2.imwrite(str(FIG / "fail_01_yaw1deg_edge_score_miss.png"), np.vstack([*panels, np.hstack(zooms)]))


def fig_fail_axis():
    """Failure 2: cùng tham số pitch/roll nhưng nuScenes có trục LiDAR khác KITTI."""
    rows = []
    for root, f in ((KITTI, "000049"), (NUSC, "scene-0103_020")):
        fr = load_frame(root, f)
        cells = []
        for kw, name in (({"pitch_deg": 2.0}, "pitch_deg=2"), ({"roll_deg": 2.0}, "roll_deg=2")):
            vis, _, _ = render(fr, perturb_extrinsic(fr["calib"], **kw), radius=3)
            vis = cv2.resize(vis, (800, int(800 * vis.shape[0] / vis.shape[1])))
            cells.append(put(vis, f"{Path(root).name} {f} {name}", scale=0.6))
        rows.append(np.hstack(cells))
    w = max(r.shape[1] for r in rows)
    cv2.imwrite(str(FIG / "fail_02_nuscenes_axis_convention.png"),
                np.vstack([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig_distance_demo()
    fig_plots()
    fig_fail_edge_miss()
    fig_fail_axis()
    print(sorted(p.name for p in FIG.iterdir()))


if __name__ == "__main__":
    main()
