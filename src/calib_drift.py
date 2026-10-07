"""Topic A — đo độ nhạy của LiDAR-camera projection với calibration drift.

Với mỗi frame và mỗi mức perturb extrinsic (chỉ thay đổi 1 yếu tố mỗi lần), đo:
  inside_fov_pct  % điểm (hữu hạn) chiếu được vào trong ảnh
  in_box_pct      % điểm thuộc object (nằm trong 3D box GT và thấy được trong ảnh với calib gốc)
                  rơi đúng vào 2D box của chính object đó, tách theo khoảng cách near/mid/far
  edge_score      alignment score: điểm LiDAR ở biên depth có nằm gần cạnh Canny của ảnh không
  edge_ratio      edge_score / edge_score của calib gốc cùng frame (1.0 = như gốc)

Ví dụ:
    python -m src.calib_drift --data-root data/kitti_mini --out results/drift_kitti.csv
    python -m src.calib_drift --data-root data/nuscenes_mini_subset --stride 4 --edge-kernel 21 --out results/drift_nuscenes.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from starter.datasets import list_frames, load_frame
from starter.projection import perturb_extrinsic, project_velo_to_image, velo_to_cam

OBJ_TYPES = {"Car", "Van", "Truck", "Bus", "Pedestrian", "Cyclist", "Bicycle", "Motorcycle"}
DIST_BINS = [("near", 0, 15), ("mid", 15, 30), ("far", 30, 1e9)]
MIN_OBJ_POINTS = 10

# (tên, tham số perturb). Mỗi cấu hình chỉ đổi đúng 1 yếu tố so với baseline.
CONFIGS = [("baseline", {})]
CONFIGS += [(f"yaw_{d}deg", {"yaw_deg": d}) for d in (0.5, 1, 2, 3)]
CONFIGS += [(f"pitch_{d}deg", {"pitch_deg": d}) for d in (0.5, 1, 2, 3)]
CONFIGS += [(f"roll_{d}deg", {"roll_deg": d}) for d in (0.5, 1, 2, 3)]
CONFIGS += [(f"tx_{int(c * 100)}cm", {"t_xyz_m": (c, 0, 0)}) for c in (0.02, 0.05, 0.10)]
CONFIGS += [(f"ty_{int(c * 100)}cm", {"t_xyz_m": (0, c, 0)}) for c in (0.02, 0.05, 0.10)]
CONFIGS += [(f"tz_{int(c * 100)}cm", {"t_xyz_m": (0, 0, c)}) for c in (0.02, 0.05, 0.10)]


def points_in_box(pts_cam: np.ndarray, obj, shrink_bottom: float = 0.15) -> np.ndarray:
    """Mask điểm (camera frame) nằm trong 3D box KITTI. Bỏ lớp 15 cm sát đáy để không lấy điểm mặt đường."""
    h, w, l = obj.dimensions
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    local = (pts_cam - obj.location) @ R            # = R^T (p - loc) cho từng hàng
    return ((np.abs(local[:, 0]) <= l / 2) & (np.abs(local[:, 2]) <= w / 2)
            & (local[:, 1] <= -shrink_bottom) & (local[:, 1] >= -h))


def edge_distance_map(image: np.ndarray) -> np.ndarray:
    """Khoảng cách (pixel) tới cạnh Canny gần nhất."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)


def edge_score(uv: np.ndarray, depth: np.ndarray, dist_map: np.ndarray,
               kernel: int = 7, jump_m: float = 1.0, sigma_px: float = 3.0) -> tuple[float, int]:
    """Alignment score kiểu Levinson & Thrun: chọn điểm foreground ở biên depth
    (có điểm khác trong cửa sổ kernel x kernel xa hơn >= jump_m), rồi lấy trung bình exp(-d/sigma),
    d = khoảng cách từ điểm tới cạnh ảnh gần nhất. Trả về (score, số điểm biên)."""
    H, W = dist_map.shape
    u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)
    far = np.zeros((H, W), np.float32)
    np.maximum.at(far, (v, u), depth.astype(np.float32))
    far = cv2.dilate(far, np.ones((kernel, kernel), np.uint8))
    is_edge = far[v, u] - depth >= jump_m
    if not is_edge.any():
        return float("nan"), 0
    return float(np.exp(-dist_map[v[is_edge], u[is_edge]] / sigma_px).mean()), int(is_edge.sum())


def dist_bin(r: float) -> str:
    return next(name for name, lo, hi in DIST_BINS if lo <= r < hi)


def run_frame(data_root: str, frame_id: str, edge_kernel: int = 7) -> list[dict]:
    fr = load_frame(data_root, frame_id)
    pts, image, calib0 = fr["points"], fr["image"], fr["calib"]
    finite = np.isfinite(pts[:, :3]).all(axis=1)
    pts = pts[finite]
    shape = image.shape
    dist_map = edge_distance_map(image)

    # Ground truth: điểm thuộc từng object, xác định 1 lần bằng calib gốc.
    # Chỉ giữ điểm chiếu được vào ảnh với calib gốc (bỏ phần object bị cắt ở mép ảnh).
    pts_cam0 = velo_to_cam(pts[:, :3], calib0)
    _, _, visible0 = project_velo_to_image(pts, calib0, shape)
    objects = []
    for obj in fr["labels"]:
        if obj.type not in OBJ_TYPES:
            continue
        m = points_in_box(pts_cam0, obj) & visible0
        if m.sum() >= MIN_OBJ_POINTS:
            objects.append((obj, m, dist_bin(float(np.linalg.norm(obj.location[[0, 2]])))))

    rows = []
    base_edge = None
    for name, kw in CONFIGS:
        calib = perturb_extrinsic(calib0, **kw)
        uv, depth, mask = project_velo_to_image(pts, calib, shape)
        score, n_edge = edge_score(uv, depth, dist_map, kernel=edge_kernel)
        if name == "baseline":
            base_edge = score

        # Điểm của object có rơi vào 2D box của nó không (cộng dồn theo bin khoảng cách)
        hit = {b[0]: [0, 0] for b in DIST_BINS}
        uv_all = np.full((len(pts), 2), np.nan)
        uv_all[mask] = uv
        for obj, m, b in objects:
            x1, y1, x2, y2 = obj.bbox
            q = uv_all[m]
            inside = ((q[:, 0] >= x1) & (q[:, 0] <= x2) & (q[:, 1] >= y1) & (q[:, 1] <= y2))
            hit[b][0] += int(inside.sum())
            hit[b][1] += len(q)
        tot_hit, tot_n = sum(h[0] for h in hit.values()), sum(h[1] for h in hit.values())

        row = {"frame": frame_id, "config": name,
               "factor": name.split("_")[0], "level": name.split("_")[1] if "_" in name else "0",
               "n_points": len(pts), "inside_fov_pct": 100 * mask.mean(),
               "n_objects": len(objects), "obj_points": tot_n,
               "in_box_pct": 100 * tot_hit / tot_n if tot_n else np.nan,
               "edge_points": n_edge, "edge_score": score,
               "edge_ratio": score / base_edge if base_edge else np.nan}
        for b, (h, n) in hit.items():
            row[f"in_box_{b}_pct"] = 100 * h / n if n else np.nan
            row[f"obj_points_{b}"] = n
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Đo ảnh hưởng của calibration drift lên projection (Topic A)")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frames", nargs="*", help="danh sách frame; bỏ trống = mọi frame")
    ap.add_argument("--stride", type=int, default=1, help="lấy 1 frame mỗi `stride` frame")
    ap.add_argument("--out", default="results/drift_kitti.csv")
    ap.add_argument("--edge-kernel", type=int, default=7,
                    help="cửa sổ (pixel) tìm điểm biên depth; LiDAR thưa (nuScenes 32 beam) cần cửa sổ lớn hơn")
    args = ap.parse_args()

    frames = args.frames or list_frames(args.data_root)[::args.stride]
    rows = []
    for f in frames:
        rows += run_frame(args.data_root, f, args.edge_kernel)
        print(f"{f}: done")
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.4f")

    # Bảng tóm tắt: cộng dồn số điểm trên mọi frame rồi mới chia (không lấy trung bình của %)
    summary = summarize(df)
    summary.to_csv(Path(args.out).with_name(Path(args.out).stem + "_summary.csv"), index=False, float_format="%.2f")
    print(summary.to_string(index=False))


def summarize(df: pd.DataFrame, edge_thresh: float = 0.9) -> pd.DataFrame:
    out = []
    for name, _ in CONFIGS:
        d = df[df.config == name]
        row = {"config": name, "frames": len(d), "inside_fov_pct": d.inside_fov_pct.mean()}
        for b, *_ in DIST_BINS:
            pct = d[f"in_box_{b}_pct"] * d[f"obj_points_{b}"] / 100
            n = d[f"obj_points_{b}"].sum()
            row[f"in_box_{b}_pct"] = 100 * pct.sum() / n if n else np.nan
        row["in_box_pct"] = 100 * (d.in_box_pct * d.obj_points / 100).sum() / d.obj_points.sum()
        row["edge_ratio_mean"] = d.edge_ratio.mean()
        row[f"detect_rate_edge<{edge_thresh}"] = 100 * (d.edge_ratio < edge_thresh).mean()
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    main()
