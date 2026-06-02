"""
Trajectory visualizer for TED.
Green: normal, Red: bad events.
"""
import os
from pathlib import Path

# keep matplotlib cache in writable temp path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

from src.agent import event_detector


# ===============================================
# config
# ================================================
OUT_DIR = REPO_ROOT / "output" / "agent"

# fixed colors for paper-like display
COLOR_TRI = "#1f77b4"
COLOR_MONSTER = "#2ca02c"
COLOR_2D = "#1f77b4"
COLOR_BAD = "#d62728"

PLOT_SMOOTH_WINDOW_FAIL = 25
PLOT_SMOOTH_WINDOW_PASS = 51
PLOT_MAX_GAP_FAIL = 3
PLOT_MAX_GAP_PASS = 8

# unified axis for compare
FIXED_2D_XLIM = (0.0, 1400.0)
FIXED_2D_YLIM = (0.0, 986.0)


# ===============================================
# tools
# ================================================
def event_intervals(events):
    intervals = []
    for e in events:
        s = int(e["start_frame"])
        t = int(e["end_frame"])
        intervals.append((s, t))
    intervals.sort()
    return intervals


def build_red_mask(frames, events):
    mask = np.zeros(len(frames), dtype=bool)
    ivs = event_intervals(events)
    if len(ivs) == 0:
        return mask

    for i, f in enumerate(frames):
        for s, t in ivs:
            if s <= int(f) <= t:
                mask[i] = True
                break
    return mask


def merge_intervals(intervals):
    if len(intervals) == 0:
        return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def red_range_text(events, fps, limit=5):
    merged = merge_intervals(event_intervals(events))
    if len(merged) == 0:
        return "none"

    parts = []
    for s, e in merged[:limit]:
        parts.append(f"{s/fps:.1f}-{e/fps:.1f}s")
    if len(merged) > limit:
        parts.append("...")
    return ", ".join(parts)


def ok_color(method: str) -> str:
    m = str(method).lower()
    if m == "monster":
        return "#2ca02c"  # green
    return "#1f77b4"      # blue (tri/2d)


def smooth_axis(values, window=31):
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    if np.sum(mask) < 3:
        return arr

    win = int(max(1, window))
    if win % 2 == 0:
        win += 1
    if win <= 1:
        return arr

    idx = np.arange(arr.size)
    interp = np.interp(idx, idx[mask], arr[mask])
    kernel = np.ones(win, dtype=float) / float(win)
    sm = np.convolve(interp, kernel, mode="same")

    out = arr.copy()
    out[mask] = sm[mask]
    return out


def robust_limits(values, q1=1.0, q2=99.0, pad_ratio=0.08):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 5:
        return None
    lo = float(np.percentile(arr, q1))
    hi = float(np.percentile(arr, q2))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi <= lo:
        return None
    pad = (hi - lo) * float(pad_ratio)
    return lo - pad, hi + pad


def build_cube_limits_from_coords(coords_list, pad_ratio=0.05, q_low=1.0, q_high=99.0):
    if coords_list is None or len(coords_list) == 0:
        return None

    all_coords = []
    for c in coords_list:
        arr = np.asarray(c, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2])
        if np.sum(mask) == 0:
            continue
        all_coords.append(arr[mask, :3])

    if len(all_coords) == 0:
        return None

    pts = np.vstack(all_coords)
    x_lo, x_hi = float(np.percentile(pts[:, 0], q_low)), float(np.percentile(pts[:, 0], q_high))
    y_lo, y_hi = float(np.percentile(pts[:, 1], q_low)), float(np.percentile(pts[:, 1], q_high))
    z_lo, z_hi = float(np.percentile(pts[:, 2], q_low)), float(np.percentile(pts[:, 2], q_high))

    cx = 0.5 * (x_lo + x_hi)
    cy = 0.5 * (y_lo + y_hi)
    cz = 0.5 * (z_lo + z_hi)

    sx = x_hi - x_lo
    sy = y_hi - y_lo
    sz = z_hi - z_lo
    side = max(sx, sy, sz)
    if side <= 1e-6:
        side = 1.0
    side = side * (1.0 + float(pad_ratio))
    half = 0.5 * side

    return {
        "xlim": (cx - half, cx + half),
        "ylim": (cy - half, cy + half),
        "zlim": (cz - half, cz + half),
    }


def detect_outlier_mask(frames, coords, valid_mask, k_out=6.0):
    n = len(frames)
    bad = np.zeros(n, dtype=bool)
    if n < 3:
        return bad

    disp = np.full(n, np.nan, dtype=float)
    for i in range(1, n):
        if not (valid_mask[i - 1] and valid_mask[i]):
            continue
        fg = max(1, int(frames[i] - frames[i - 1]))
        disp[i] = float(np.linalg.norm(coords[i] - coords[i - 1]) / float(fg))

    fin = disp[np.isfinite(disp)]
    if fin.size < 10:
        return bad

    med = float(np.median(fin))
    mad = float(np.median(np.abs(fin - med)))
    sigma = float(1.4826 * mad)
    if sigma < 1e-9:
        sigma = float(np.std(fin)) + 1e-9
    thr1 = med + float(k_out) * sigma
    thr2 = float(np.percentile(fin, 99)) * 1.5
    thr = max(thr1, thr2)

    out_count = 0
    for i in range(1, n):
        if np.isfinite(disp[i]) and disp[i] > thr:
            bad[i] = True
            out_count += 1

    if out_count > int(0.15 * max(1, fin.size)):
        return np.zeros(n, dtype=bool)

    return bad


def detect_coord_outlier_mask(coords, valid_mask, k_out=7.0):
    coords = np.asarray(coords, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    n = len(valid_mask)
    bad = np.zeros(n, dtype=bool)

    if coords.ndim != 2 or coords.shape[0] == 0:
        return bad

    for d in range(coords.shape[1]):
        arr = coords[:, d]
        fin = arr[valid_mask & np.isfinite(arr)]
        if fin.size < 20:
            continue

        med = float(np.median(fin))
        mad = float(np.median(np.abs(fin - med)))
        sigma = float(1.4826 * mad)
        if sigma < 1e-9:
            sigma = float(np.std(fin)) + 1e-9
        lo = med - float(k_out) * sigma
        hi = med + float(k_out) * sigma

        bad = bad | (np.isfinite(arr) & ((arr < lo) | (arr > hi)))

    if np.sum(bad) > int(0.20 * max(1, np.sum(valid_mask))):
        return np.zeros(n, dtype=bool)
    return bad


def detect_percentile_outlier_mask(coords, valid_mask, q_low=0.5, q_high=99.5):
    coords = np.asarray(coords, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    n = len(valid_mask)
    bad = np.zeros(n, dtype=bool)

    if coords.ndim != 2 or coords.shape[0] == 0:
        return bad

    for d in range(coords.shape[1]):
        arr = coords[:, d]
        fin = arr[valid_mask & np.isfinite(arr)]
        if fin.size < 20:
            continue

        lo = float(np.percentile(fin, q_low))
        hi = float(np.percentile(fin, q_high))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue

        bad = bad | (np.isfinite(arr) & ((arr < lo) | (arr > hi)))

    if np.sum(bad) > int(0.35 * max(1, np.sum(valid_mask))):
        return np.zeros(n, dtype=bool)
    return bad


def detect_radius_outlier_mask(coords, valid_mask, k_out=6.0):
    coords = np.asarray(coords, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    n = len(valid_mask)
    bad = np.zeros(n, dtype=bool)

    if coords.ndim != 2 or coords.shape[0] == 0:
        return bad

    vv = coords[valid_mask]
    vv = vv[np.all(np.isfinite(vv), axis=1)]
    if vv.shape[0] < 20:
        return bad

    center = np.median(vv, axis=0)
    dist = np.linalg.norm(vv - center, axis=1)
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med)))
    sigma = float(1.4826 * mad)
    if sigma < 1e-9:
        sigma = float(np.std(dist)) + 1e-9
    thr = med + float(k_out) * sigma

    all_dist = np.full(n, np.nan, dtype=float)
    finite_all = np.all(np.isfinite(coords), axis=1)
    all_dist[finite_all] = np.linalg.norm(coords[finite_all] - center, axis=1)
    bad = np.isfinite(all_dist) & (all_dist > thr)

    if np.sum(bad) > int(0.30 * max(1, np.sum(valid_mask))):
        return np.zeros(n, dtype=bool)
    return bad


def segment_break_threshold(frames, coords, finite_mask, k=3.0):
    disp = []
    for i in range(1, len(frames)):
        if int(frames[i] - frames[i - 1]) > 1:
            continue
        if not (finite_mask[i - 1] and finite_mask[i]):
            continue
        d = float(np.linalg.norm(coords[i] - coords[i - 1]))
        if np.isfinite(d):
            disp.append(d)

    arr = np.asarray(disp, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 10:
        return np.inf

    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = float(1.4826 * mad)
    if sigma < 1e-9:
        sigma = float(np.std(arr)) + 1e-9
    return med + float(k) * sigma


def build_segment_ids(frames, coords, finite_mask, break_thr):
    n = len(frames)
    seg_ids = np.full(n, -1, dtype=int)
    if n == 0:
        return seg_ids

    seg = 0
    seg_ids[0] = seg
    for i in range(1, n):
        is_break = False
        if int(frames[i] - frames[i - 1]) > 1:
            is_break = True
        elif not (finite_mask[i - 1] and finite_mask[i]):
            is_break = True
        else:
            d = float(np.linalg.norm(coords[i] - coords[i - 1]))
            if np.isfinite(break_thr) and d > break_thr:
                is_break = True

        if is_break:
            seg += 1
        seg_ids[i] = seg

    return seg_ids


def build_full_track(frames, coords, keep_mask, max_fill_gap=2):
    frames = np.asarray(frames, dtype=int)
    coords = np.asarray(coords, dtype=float)
    keep_mask = np.asarray(keep_mask, dtype=bool)

    keep_idx = np.where(keep_mask)[0]
    if len(keep_idx) == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float), np.asarray([], dtype=bool)
    if len(keep_idx) == 1:
        i = int(keep_idx[0])
        return np.asarray([int(frames[i])], dtype=int), np.asarray([coords[i]], dtype=float), np.asarray([False], dtype=bool)

    out_frames = [int(frames[int(keep_idx[0])])]
    out_coords = [np.asarray(coords[int(keep_idx[0])], dtype=float)]
    out_fill = [False]

    fill_gap = int(max(0, max_fill_gap))

    for j in range(1, len(keep_idx)):
        i0 = int(keep_idx[j - 1])
        i1 = int(keep_idx[j])
        f0 = int(frames[i0])
        f1 = int(frames[i1])
        c0 = np.asarray(coords[i0], dtype=float)
        c1 = np.asarray(coords[i1], dtype=float)

        gap = f1 - f0
        if gap > 1 and gap <= (fill_gap + 1):
            for g in range(1, gap):
                a = float(g) / float(gap)
                ff = f0 + g
                cc = (1.0 - a) * c0 + a * c1
                out_frames.append(int(ff))
                out_coords.append(np.asarray(cc, dtype=float))
                out_fill.append(True)

        out_frames.append(f1)
        out_coords.append(c1)
        out_fill.append(False)

    return np.asarray(out_frames, dtype=int), np.asarray(out_coords, dtype=float), np.asarray(out_fill, dtype=bool)


# align 3d track to reference
# ================================================
# ===============================================
# plot core
# ================================================
def plot_2d_track(
    track,
    events,
    case_id,
    method,
    fps,
    out_png,
    kpt_id="",
    side="",
    k_break=3.0,
    smooth_window=31,
    max_gap=2,
    gate_pass=False,
    axis_2d=None,
):
    frames = np.asarray(track["frames"])
    coords = np.asarray(track["coords"])
    if len(frames) < 2:
        return None

    x = smooth_axis(coords[:, 0], window=smooth_window)
    y = smooth_axis(coords[:, 1], window=smooth_window)
    plot_coords = np.column_stack([x, y])
    valid_plot = np.isfinite(x) & np.isfinite(y)

    k_step = max(2.8, k_break - 0.2) if gate_pass else max(3.0, k_break)
    k_axis = 5.5 if gate_pass else max(6.0, k_break)

    outlier_step = detect_outlier_mask(frames, plot_coords, valid_plot, k_out=k_step)
    outlier_axis = detect_coord_outlier_mask(plot_coords, valid_plot, k_out=k_axis)
    outlier_range = detect_percentile_outlier_mask(plot_coords, valid_plot, q_low=0.5, q_high=99.5)
    outlier_radius = detect_radius_outlier_mask(plot_coords, valid_plot, k_out=5.0 if gate_pass else 6.0)

    keep_mask = valid_plot & (~outlier_step) & (~outlier_axis) & (~outlier_range) & (~outlier_radius)
    valid_count = int(np.sum(valid_plot))
    keep_count = int(np.sum(keep_mask))
    if valid_count > 0 and keep_count < int(0.60 * valid_count):
        keep_mask = valid_plot & (~outlier_axis) & (~outlier_range)
    if valid_count > 0 and int(np.sum(keep_mask)) < int(0.40 * valid_count):
        keep_mask = valid_plot & (~outlier_range)

    full_frames, full_coords, filled_mask = build_full_track(frames, plot_coords, keep_mask, max_fill_gap=max_gap)
    if len(full_frames) < 2:
        return None
    xf = full_coords[:, 0]
    yf = full_coords[:, 1]
    red_mask = build_red_mask(full_frames, events)
    finite = np.isfinite(xf) & np.isfinite(yf)
    seg_break_thr = segment_break_threshold(full_frames, full_coords, finite, k=2.8 if gate_pass else 3.2)
    seg_ids = build_segment_ids(full_frames, full_coords, finite, seg_break_thr)
    keep_seg = set(np.unique(seg_ids).tolist())
    if gate_pass and len(keep_seg) > 1:
        counts = {}
        for sid in keep_seg:
            counts[int(sid)] = int(np.sum((seg_ids == sid) & finite))
        max_count = max(counts.values()) if counts else 0
        min_keep = max(12, int(0.35 * max_count))
        keep_seg = {sid for sid, c in counts.items() if c >= min_keep}

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    color_ok = ok_color(method)

    for i in range(1, len(full_frames)):
        if int(full_frames[i] - full_frames[i - 1]) > 1:
            continue
        if not (finite[i - 1] and finite[i]):
            continue
        if seg_ids[i] != seg_ids[i - 1]:
            continue
        if int(seg_ids[i]) not in keep_seg:
            continue
        seg_disp = float(np.linalg.norm(full_coords[i] - full_coords[i - 1]))
        if np.isfinite(seg_break_thr) and seg_disp > seg_break_thr:
            continue
        is_red = bool(red_mask[i - 1] or red_mask[i])
        is_fill = bool(filled_mask[i - 1] or filled_mask[i])

        ls = "--" if is_fill else "-"
        lw = 1.2 if is_fill else 1.6
        cc = COLOR_BAD if is_red else color_ok

        ax.plot([xf[i - 1], xf[i]], [yf[i - 1], yf[i]], color=cc, linestyle=ls, linewidth=lw, alpha=0.95, zorder=2)

    red_pts = red_mask & finite & np.isin(seg_ids, list(keep_seg))
    if np.any(red_pts):
        ax.scatter(xf[red_pts], yf[red_pts], s=10, c=COLOR_BAD, alpha=0.95, zorder=4)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    if axis_2d is None:
        xlim = FIXED_2D_XLIM
        ylim = FIXED_2D_YLIM
    else:
        xlim = axis_2d.get("xlim", FIXED_2D_XLIM)
        ylim = axis_2d.get("ylim", FIXED_2D_YLIM)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    red_txt = red_range_text(events, fps)
    ax.set_title(f"case={case_id} | {side} | {method} | kpt={kpt_id} | fps={fps} | red: {red_txt}")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_png


def plot_3d_track(
    track,
    events,
    case_id,
    method,
    fps,
    out_png,
    kpt_id="",
    side="",
    k_break=3.0,
    smooth_window=31,
    max_gap=2,
    gate_pass=False,
    axis_3d=None,
):
    frames = np.asarray(track["frames"])
    coords = np.asarray(track["coords"])
    if len(frames) < 2 or coords.shape[1] < 3:
        return None

    x = smooth_axis(coords[:, 0], window=smooth_window)
    y = smooth_axis(coords[:, 1], window=smooth_window)
    z = smooth_axis(coords[:, 2], window=smooth_window)
    plot_coords = np.column_stack([x, y, z])
    valid_plot = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    k_step = max(2.8, k_break - 0.2) if gate_pass else max(3.0, k_break)
    k_axis = 5.5 if gate_pass else max(6.0, k_break)

    outlier_step = detect_outlier_mask(frames, plot_coords, valid_plot, k_out=k_step)
    outlier_axis = detect_coord_outlier_mask(plot_coords, valid_plot, k_out=k_axis)
    outlier_range = detect_percentile_outlier_mask(plot_coords, valid_plot, q_low=0.5, q_high=99.5)
    outlier_radius = detect_radius_outlier_mask(plot_coords, valid_plot, k_out=5.0 if gate_pass else 6.0)

    keep_mask = valid_plot & (~outlier_step) & (~outlier_axis) & (~outlier_range) & (~outlier_radius)
    valid_count = int(np.sum(valid_plot))
    keep_count = int(np.sum(keep_mask))
    if valid_count > 0 and keep_count < int(0.60 * valid_count):
        keep_mask = valid_plot & (~outlier_axis) & (~outlier_range)
    if valid_count > 0 and int(np.sum(keep_mask)) < int(0.40 * valid_count):
        keep_mask = valid_plot & (~outlier_range)

    full_frames, full_coords, filled_mask = build_full_track(frames, plot_coords, keep_mask, max_fill_gap=max_gap)
    if len(full_frames) < 2:
        return None
    xf = full_coords[:, 0]
    yf = full_coords[:, 1]
    zf = full_coords[:, 2]
    red_mask = build_red_mask(full_frames, events)
    finite = np.isfinite(xf) & np.isfinite(yf) & np.isfinite(zf)
    seg_break_thr = segment_break_threshold(full_frames, full_coords, finite, k=2.8 if gate_pass else 3.2)
    seg_ids = build_segment_ids(full_frames, full_coords, finite, seg_break_thr)
    keep_seg = set(np.unique(seg_ids).tolist())
    if gate_pass and len(keep_seg) > 1:
        counts = {}
        for sid in keep_seg:
            counts[int(sid)] = int(np.sum((seg_ids == sid) & finite))
        max_count = max(counts.values()) if counts else 0
        min_keep = max(12, int(0.35 * max_count))
        keep_seg = {sid for sid, c in counts.items() if c >= min_keep}

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=22, azim=-60)
    ax.set_proj_type("ortho")
    color_ok = ok_color(method)

    for i in range(1, len(full_frames)):
        if int(full_frames[i] - full_frames[i - 1]) > 1:
            continue
        if not (finite[i - 1] and finite[i]):
            continue
        if seg_ids[i] != seg_ids[i - 1]:
            continue
        if int(seg_ids[i]) not in keep_seg:
            continue
        seg_disp = float(np.linalg.norm(full_coords[i] - full_coords[i - 1]))
        if np.isfinite(seg_break_thr) and seg_disp > seg_break_thr:
            continue
        is_red = bool(red_mask[i - 1] or red_mask[i])
        is_fill = bool(filled_mask[i - 1] or filled_mask[i])

        ls = "--" if is_fill else "-"
        lw = 1.2 if is_fill else 1.6
        cc = COLOR_BAD if is_red else color_ok

        ax.plot([xf[i - 1], xf[i]], [yf[i - 1], yf[i]], [zf[i - 1], zf[i]], color=cc, linestyle=ls, linewidth=lw, alpha=0.95, zorder=2)

    red_pts = red_mask & finite & np.isin(seg_ids, list(keep_seg))
    if np.any(red_pts):
        ax.scatter(xf[red_pts], yf[red_pts], zf[red_pts], s=12, c=COLOR_BAD, alpha=0.95, depthshade=False, zorder=4)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if axis_3d is None:
        cube_lim = build_cube_limits_from_coords([full_coords], pad_ratio=0.05)
    else:
        cube_lim = axis_3d

    if cube_lim is not None:
        ax.set_xlim(*cube_lim["xlim"])
        ax.set_ylim(*cube_lim["ylim"])
        ax.set_zlim(*cube_lim["zlim"])
    ax.set_box_aspect((1.0, 1.0, 1.0))

    red_txt = red_range_text(events, fps)
    ax.set_title(f"case={case_id} | {side} | {method} | kpt={kpt_id} | fps={fps} | red: {red_txt}")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_png


def plot_timeline_disp(track, events, case_id, method, fps, out_png, kpt_id="", side=""):
    metrics = event_detector.compute_track_metrics(track, fps=fps)
    frames = metrics["frames"]
    if len(frames) == 0:
        return None

    t = frames / float(fps)
    disp = metrics.get("disp", np.asarray([]))

    fig, ax = plt.subplots(1, 1, figsize=(10, 3.2))

    if len(disp) == len(t):
        ax.plot(t, disp, color="#1f77b4", linewidth=1.1)
    ax.set_ylabel("disp")
    ax.set_xlabel("time (s)")
    ax.grid(True, alpha=0.25)

    for s, e in merge_intervals(event_intervals(events)):
        s_t = s / float(fps)
        e_t = e / float(fps)
        ax.axvspan(s_t, e_t, color=COLOR_BAD, alpha=0.14)

    red_txt = red_range_text(events, fps)
    fig.suptitle(f"case={case_id} | {side} | {method} | kpt={kpt_id} | fps={fps} | red: {red_txt}")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_png


# ===============================================
# public api
# ================================================
def save_method_figures(det_result, out_dir=OUT_DIR, save_timeline=True, k_break=None, axis_2d=None, axis_3d=None):
    case_id = det_result["case_id"]
    method = det_result["method"]
    track = det_result["track"]
    events = det_result["events"]
    fps = float(det_result.get("fps", 30))
    kpt_id = str(det_result.get("kpt_id", ""))
    side = str(det_result.get("view_side", ""))
    gate_pass = bool(det_result.get("gate_pass", False))

    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_files = []

    k_break_value = 6.0 if k_break is None else float(k_break)
    if str(method).lower() == "2d":
        k_break_value = k_break_value + 0.5

    if gate_pass:
        smooth_window = PLOT_SMOOTH_WINDOW_PASS
        max_gap = PLOT_MAX_GAP_PASS
    else:
        smooth_window = PLOT_SMOOTH_WINDOW_FAIL
        max_gap = PLOT_MAX_GAP_FAIL

    suffix = f"{case_id}_{side}_{method}" if side != "" else f"{case_id}_{method}"
    traj_png = fig_dir / f"{suffix}_traj.png"

    if track.get("dim", 2) >= 3:
        p = plot_3d_track(
            track,
            events,
            case_id,
            method,
            fps,
            traj_png,
            kpt_id=kpt_id,
            side=side,
            k_break=k_break_value,
            smooth_window=smooth_window,
            max_gap=max_gap,
            gate_pass=gate_pass,
            axis_3d=axis_3d,
        )
    else:
        p = plot_2d_track(
            track,
            events,
            case_id,
            method,
            fps,
            traj_png,
            kpt_id=kpt_id,
            side=side,
            k_break=k_break_value,
            smooth_window=smooth_window,
            max_gap=max_gap,
            gate_pass=gate_pass,
            axis_2d=axis_2d,
        )

    if p is not None:
        out_files.append(str(p))

    if save_timeline:
        timeline_png = fig_dir / f"{suffix}_timeline.png"
        p2 = plot_timeline_disp(track, events, case_id, method, fps, timeline_png, kpt_id=kpt_id, side=side)
        if p2 is not None:
            out_files.append(str(p2))

    return out_files
