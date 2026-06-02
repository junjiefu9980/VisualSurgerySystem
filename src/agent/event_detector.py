"""
Temporal Event Detector (TED).
Detect bad trajectory segments from 2D/3D track.
"""
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


# ===============================================
# config
# ================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
DET_DIR = REPO_ROOT / "output" / "detections"

DEFAULT_FPS = 30
# default robust threshold: median + 6 * MAD (strong outlier gate, reproducible)
DEFAULT_K_MAD = 6.0
DEFAULT_MIN_LEN = 5
DEFAULT_VIEW_SIDE = "left"

DEFAULT_METHOD_CSV = {
    "2d": DET_DIR / "2d_results.csv",
    "tri": DET_DIR / "3d_tri_results.csv",
    "monster": DET_DIR / "3d_monster_results.csv",
}

EVENT_TYPES = [
    "JITTER_SPIKE",
    "JUMP_OUTLIER",
    "STALL_STOP",
    "MISSING_LOW_VALID",
]


# ===============================================
# basic tools
# ================================================
def safe_float(x, default=None):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def safe_int(x, default=None):
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except (TypeError, ValueError):
        return default


def normalize_case_id(case_id):
    s = str(case_id).strip()
    if s.isdigit():
        return str(int(s))
    return s


def case_equal(a, b):
    sa = str(a).strip()
    sb = str(b).strip()
    if sa.isdigit() and sb.isdigit():
        return int(sa) == int(sb)
    return sa == sb


def parse_filename(filename):
    stem = Path(str(filename)).stem
    m = re.match(r"^(\d+)_([a-zA-Z]+)_video_(\d+)$", stem)
    if m:
        case_id = str(int(m.group(1)))
        side = m.group(2).lower()
        frame_idx = int(m.group(3))
        return case_id, side, frame_idx

    parts = stem.split("_")
    case_id = None
    frame_idx = None

    if len(parts) >= 2 and parts[0].isdigit() and parts[-1].isdigit():
        case_id = str(int(parts[0]))
        frame_idx = int(parts[-1])

    text = stem.lower()
    side = "left" if "left" in text else "right" if "right" in text else ""
    return case_id, side, frame_idx


def robust_stats(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {
            "median": np.nan,
            "mad": np.nan,
            "sigma": np.nan,
            "p95": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = float(1.4826 * mad)
    p95 = float(np.percentile(arr, 95))

    # stable fallback when MAD too small
    if sigma < 1e-9:
        q75 = float(np.percentile(arr, 75))
        q25 = float(np.percentile(arr, 25))
        sigma = max((q75 - q25) / 1.349, 1e-9)

    return {
        "median": med,
        "mad": mad,
        "sigma": sigma,
        "p95": p95,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def robust_high_threshold(values, k_mad):
    st = robust_stats(values)
    if not np.isfinite(st["median"]) or not np.isfinite(st["sigma"]):
        return np.nan, st
    thr = st["median"] + float(k_mad) * st["sigma"]
    return float(thr), st


def robust_low_threshold(values, k_mad):
    st = robust_stats(values)
    if not np.isfinite(st["median"]) or not np.isfinite(st["sigma"]):
        return np.nan, st
    thr = st["median"] - float(k_mad) * st["sigma"]
    return float(thr), st


def severity_level(score):
    if score is None or not np.isfinite(score):
        return "low"
    if score < 3:
        return "low"
    if score < 6:
        return "med"
    return "high"


# ===============================================
# schema parse
# ================================================
def find_first(headers, names):
    hset = {h.lower(): h for h in headers}
    for n in names:
        if n.lower() in hset:
            return hset[n.lower()]
    return None


def detect_dimension(headers, method):
    # method fixed dimension
    if method == "2d":
        return 2
    if method in ("tri", "monster"):
        return 3
    for h in headers:
        if str(h).endswith("_x3d"):
            return 3
    return 2


def detect_wide_kpt_candidates(headers, dim):
    cands = []
    header_set = set(headers)

    if dim == 3:
        for h in headers:
            m = re.match(r"^([A-Za-z0-9]+)_x3d$", h)
            if not m:
                continue
            name = m.group(1)
            if f"{name}_y3d" in header_set and f"{name}_z3d" in header_set:
                cands.append(name)
    else:
        for h in headers:
            m = re.match(r"^([A-Za-z0-9]+)_x$", h)
            if not m:
                continue
            name = m.group(1)
            if "bbox" in name.lower():
                continue
            if f"{name}_y" in header_set:
                cands.append(name)

    cands = sorted(set(cands))
    return cands


def infer_schema(headers, method):
    dim = detect_dimension(headers, method)

    kpt_col = find_first(headers, ["kpt_id", "kp_id", "keypoint_id"]) 

    if dim == 3:
        x_col = find_first(headers, ["x3d_kf", "coord_x3d_kf", "x3d", "coord_x3d", "x_kf", "x", "coord_x"])
        y_col = find_first(headers, ["y3d_kf", "coord_y3d_kf", "y3d", "coord_y3d", "y_kf", "y", "coord_y"])
        z_col = find_first(headers, ["z3d_kf", "coord_z3d_kf", "z3d", "coord_z3d", "z_kf", "z", "coord_z", "depth"]) 
    else:
        x_col = find_first(headers, ["x_kf", "coord_x_kf", "x", "coord_x", "u", "px"])
        y_col = find_first(headers, ["y_kf", "coord_y_kf", "y", "coord_y", "v", "py"])
        z_col = None

    valid_col = find_first(headers, ["valid", "is_valid", "kp_valid", "visible"]) 
    conf_col = find_first(headers, ["conf", "score", "probability", "kp_conf"]) 

    generic_long_ok = (kpt_col is not None and x_col is not None and y_col is not None)
    wide_kpts = detect_wide_kpt_candidates(headers, dim)

    if generic_long_ok:
        fmt = "long"
    else:
        fmt = "wide"

    return {
        "dim": dim,
        "format": fmt,
        "kpt_col": kpt_col,
        "x_col": x_col,
        "y_col": y_col,
        "z_col": z_col,
        "valid_col": valid_col,
        "conf_col": conf_col,
        "wide_kpts": wide_kpts,
        "header_set": set(headers),
    }


# ===============================================
# load rows
# ================================================
def load_case_rows(csv_path, case_id, view_side=DEFAULT_VIEW_SIDE):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"missing csv: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])

        case_col = find_first(headers, ["case_id", "case", "video_id"])
        frame_col = find_first(headers, ["frame", "frame_id", "frame_idx", "idx"])
        side_col = find_first(headers, ["side", "camera", "view"]) 
        file_col = find_first(headers, ["filename", "file", "image", "image_name"])

        for row in reader:
            row = dict(row)

            row_case = None
            row_side = ""
            row_frame = None

            if case_col is not None:
                c = str(row.get(case_col, "")).strip()
                if c != "":
                    row_case = normalize_case_id(c)

            if frame_col is not None:
                row_frame = safe_int(row.get(frame_col), default=None)

            if side_col is not None:
                row_side = str(row.get(side_col, "")).strip().lower()

            if file_col is not None:
                p_case, p_side, p_frame = parse_filename(row.get(file_col, ""))
                if row_case is None and p_case is not None:
                    row_case = p_case
                if row_frame is None and p_frame is not None:
                    row_frame = p_frame
                if row_side == "" and p_side is not None:
                    row_side = p_side

            if row_case is None or row_frame is None:
                continue
            if not case_equal(row_case, case_id):
                continue

            if view_side in ("left", "right") and row_side in ("left", "right"):
                if row_side != view_side:
                    continue

            row["_case_id"] = normalize_case_id(row_case)
            row["_frame_idx"] = int(row_frame)
            row["_side"] = row_side
            rows.append(row)

    return headers, rows


# ===============================================
# kpt choose
# ================================================
def build_spec_for_wide(schema, kpt_name):
    dim = schema["dim"]
    header_set = schema.get("header_set", set())

    if dim == 3:
        x_raw = f"{kpt_name}_x3d"
        y_raw = f"{kpt_name}_y3d"
        z_raw = f"{kpt_name}_z3d"
        x_col = f"{x_raw}_kf" if f"{x_raw}_kf" in header_set else x_raw
        y_col = f"{y_raw}_kf" if f"{y_raw}_kf" in header_set else y_raw
        z_col = f"{z_raw}_kf" if f"{z_raw}_kf" in header_set else z_raw
    else:
        x_raw = f"{kpt_name}_x"
        y_raw = f"{kpt_name}_y"
        x_col = f"{x_raw}_kf" if f"{x_raw}_kf" in header_set else x_raw
        y_col = f"{y_raw}_kf" if f"{y_raw}_kf" in header_set else y_raw
        z_col = None

    if x_col not in header_set or y_col not in header_set:
        return None
    if dim == 3 and z_col not in header_set:
        return None

    conf_col = f"{kpt_name}_conf"
    if conf_col not in header_set:
        if str(kpt_name).startswith("L") and "L_conf" in header_set:
            conf_col = "L_conf"
        elif str(kpt_name).startswith("R") and "R_conf" in header_set:
            conf_col = "R_conf"

    return {
        "format": "wide",
        "kpt_name": kpt_name,
        "dim": dim,
        "x_col": x_col,
        "y_col": y_col,
        "z_col": z_col,
        "conf_col": conf_col,
        "valid_col": schema["valid_col"],
    }


def build_spec_for_long(schema, kpt_value):
    return {
        "format": "long",
        "kpt_value": str(kpt_value),
        "dim": schema["dim"],
        "kpt_col": schema["kpt_col"],
        "x_col": schema["x_col"],
        "y_col": schema["y_col"],
        "z_col": schema["z_col"],
        "conf_col": schema["conf_col"],
        "valid_col": schema["valid_col"],
    }


def extract_point(row, spec):
    if spec["format"] == "long":
        kcol = spec["kpt_col"]
        if kcol is None:
            return None
        if str(row.get(kcol, "")).strip() != str(spec["kpt_value"]):
            return None

    x = safe_float(row.get(spec["x_col"], None), default=None)
    y = safe_float(row.get(spec["y_col"], None), default=None)

    if spec["dim"] == 3:
        z = safe_float(row.get(spec["z_col"], None), default=None)
        coords = [x, y, z]
    else:
        coords = [x, y]

    conf = None
    conf_col = spec.get("conf_col")
    if conf_col and conf_col in row:
        conf = safe_float(row.get(conf_col), default=None)

    valid_col = spec.get("valid_col")
    valid_by_flag = True
    if valid_col and valid_col in row:
        vflag = safe_float(row.get(valid_col), default=None)
        if vflag is not None and vflag <= 0:
            valid_by_flag = False

    finite = all(v is not None and np.isfinite(v) for v in coords)
    valid = bool(finite and valid_by_flag)
    if conf is not None and np.isfinite(conf):
        valid = valid and (conf > 0.0)

    return {
        "frame_idx": int(row["_frame_idx"]),
        "coords": np.asarray(coords, dtype=float) if finite else np.asarray([np.nan] * len(coords), dtype=float),
        "valid": bool(valid),
        "conf": conf,
        "row": row,
    }


def choose_kpt_spec(schema, rows, forced_kpt=None):
    if not rows:
        return None

    fk = str(forced_kpt).strip().lower()

    # forced side auto
    if schema["format"] == "wide" and fk in ("auto_left", "left", "auto_l", "l"):
        cands = [k for k in schema["wide_kpts"] if str(k).startswith("L")]
    elif schema["format"] == "wide" and fk in ("auto_right", "right", "auto_r", "r"):
        cands = [k for k in schema["wide_kpts"] if str(k).startswith("R")]
    else:
        cands = None

    if cands is not None:
        if not cands:
            return None
        best = None
        best_score = (-1, -1.0)
        for name in cands:
            spec = build_spec_for_wide(schema, name)
            if spec is None:
                continue
            valid_count = 0
            conf_sum = 0.0
            conf_n = 0
            for r in rows:
                pt = extract_point(r, spec)
                if pt is None:
                    continue
                if pt["valid"]:
                    valid_count += 1
                if pt["conf"] is not None and np.isfinite(pt["conf"]):
                    conf_sum += float(pt["conf"])
                    conf_n += 1
            conf_mean = conf_sum / conf_n if conf_n > 0 else 0.0
            score = (valid_count, conf_mean)
            if score > best_score:
                best = spec
                best_score = score
        return best

    # forced fixed kpt
    if forced_kpt not in (None, "", "auto"):
        if schema["format"] == "wide":
            spec = build_spec_for_wide(schema, str(forced_kpt))
            if spec is not None:
                return spec
        elif schema["format"] == "long":
            return build_spec_for_long(schema, str(forced_kpt))

    # long format
    if schema["format"] == "long":
        kcol = schema["kpt_col"]
        uniq = sorted({str(r.get(kcol, "")).strip() for r in rows if str(r.get(kcol, "")).strip() != ""})
        if not uniq:
            return None

        best = None
        best_score = (-1, -1.0)
        for kid in uniq:
            spec = build_spec_for_long(schema, kid)
            valid_count = 0
            conf_sum = 0.0
            conf_n = 0
            for r in rows:
                pt = extract_point(r, spec)
                if pt is None:
                    continue
                if pt["valid"]:
                    valid_count += 1
                if pt["conf"] is not None and np.isfinite(pt["conf"]):
                    conf_sum += float(pt["conf"])
                    conf_n += 1
            conf_mean = conf_sum / conf_n if conf_n > 0 else 0.0
            score = (valid_count, conf_mean)
            if score > best_score:
                best = spec
                best_score = score
        return best

    # wide format
    cands = schema["wide_kpts"]
    if not cands:
        return None

    best = None
    best_score = (-1, -1.0)
    for name in cands:
        spec = build_spec_for_wide(schema, name)
        if spec is None:
            continue
        valid_count = 0
        conf_sum = 0.0
        conf_n = 0
        for r in rows:
            pt = extract_point(r, spec)
            if pt is None:
                continue
            if pt["valid"]:
                valid_count += 1
            if pt["conf"] is not None and np.isfinite(pt["conf"]):
                conf_sum += float(pt["conf"])
                conf_n += 1
        conf_mean = conf_sum / conf_n if conf_n > 0 else 0.0
        score = (valid_count, conf_mean)
        if score > best_score:
            best = spec
            best_score = score

    return best


# ===============================================
# build track
# ================================================
def build_track(rows, spec, continuity=True, k_gate=6.0, window=30):
    frame_cands = {}

    for r in rows:
        pt = extract_point(r, spec)
        if pt is None:
            continue

        fidx = pt["frame_idx"]
        quality = 0.0
        if pt["valid"]:
            quality += 10.0
        if pt["conf"] is not None and np.isfinite(pt["conf"]):
            quality += float(pt["conf"])

        frame_cands.setdefault(fidx, []).append(
            {
                "frame_idx": fidx,
                "coords": pt["coords"],
                "valid": pt["valid"],
                "conf": pt["conf"],
                "quality": quality,
            }
        )

    if not frame_cands:
        return {
            "frames": np.asarray([], dtype=int),
            "coords": np.asarray([], dtype=float),
            "valid": np.asarray([], dtype=bool),
            "conf": np.asarray([], dtype=float),
            "switch_flag": np.asarray([], dtype=bool),
            "dim": spec["dim"],
        }

    frame_keys = sorted(frame_cands.keys())
    chosen = []
    disp_hist = []
    prev_coords = None
    prev_valid = False
    prev_frame = None

    for fidx in frame_keys:
        cands = frame_cands.get(fidx, [])
        if len(cands) == 0:
            continue

        best_quality = max(cands, key=lambda x: float(x.get("quality", 0.0)))
        pick = best_quality
        switch_flag = False

        if continuity and prev_valid and prev_coords is not None and np.all(np.isfinite(prev_coords)):
            finite_cands = []
            for c in cands:
                cc = np.asarray(c.get("coords"), dtype=float)
                if bool(c.get("valid", False)) and np.all(np.isfinite(cc)):
                    finite_cands.append(c)

            if len(finite_cands) > 0:
                dists = []
                for c in finite_cands:
                    cc = np.asarray(c["coords"], dtype=float)
                    fg = max(1, int(fidx - int(prev_frame)))
                    dists.append(float(np.linalg.norm(cc - prev_coords)) / float(fg))

                gate = np.inf
                hist = np.asarray(disp_hist[-int(max(5, window)):], dtype=float)
                hist = hist[np.isfinite(hist)]
                if hist.size >= 2:
                    st = robust_stats(hist)
                    if np.isfinite(st["median"]) and np.isfinite(st["sigma"]):
                        gate = float(st["median"] + float(k_gate) * float(st["sigma"]))

                idx = int(np.argmin(np.asarray(dists, dtype=float)))
                nearest = finite_cands[idx]
                nearest_dist = float(dists[idx])

                if np.isfinite(gate) and nearest_dist <= gate:
                    pick = nearest
                    if pick is not best_quality:
                        switch_flag = True
                else:
                    pick = best_quality
                    if nearest is not best_quality:
                        switch_flag = True

        if prev_valid:
            pc = np.asarray(pick.get("coords"), dtype=float)
            if bool(pick.get("valid", False)) and np.all(np.isfinite(pc)):
                fg = max(1, int(fidx - int(prev_frame)))
                disp_hist.append(float(np.linalg.norm(pc - prev_coords)) / float(fg))

        if bool(pick.get("valid", False)) and np.all(np.isfinite(np.asarray(pick.get("coords"), dtype=float))):
            prev_coords = np.asarray(pick.get("coords"), dtype=float)
            prev_valid = True
            prev_frame = int(fidx)
        else:
            prev_valid = False
            prev_frame = int(fidx)

        out_row = dict(pick)
        out_row["switch_flag"] = bool(switch_flag)
        chosen.append(out_row)

    if len(chosen) == 0:
        return {
            "frames": np.asarray([], dtype=int),
            "coords": np.asarray([], dtype=float),
            "valid": np.asarray([], dtype=bool),
            "conf": np.asarray([], dtype=float),
            "switch_flag": np.asarray([], dtype=bool),
            "dim": spec["dim"],
        }

    frames = np.asarray([int(x["frame_idx"]) for x in chosen], dtype=int)
    coords = np.asarray([x["coords"] for x in chosen], dtype=float)
    valid = np.asarray([bool(x["valid"]) for x in chosen], dtype=bool)
    conf = np.asarray(
        [np.nan if x["conf"] is None else float(x["conf"]) for x in chosen],
        dtype=float,
    )
    switch_flag = np.asarray([bool(x.get("switch_flag", False)) for x in chosen], dtype=bool)

    return {
        "frames": frames,
        "coords": coords,
        "valid": valid,
        "conf": conf,
        "switch_flag": switch_flag,
        "dim": spec["dim"],
    }


# ===============================================
# metrics
# ================================================
def compute_track_metrics(track, fps):
    frames = track["frames"]
    coords = track["coords"]
    valid = track["valid"]

    n = len(frames)
    disp = np.full(n, np.nan, dtype=float)
    speed = np.full(n, np.nan, dtype=float)
    acc = np.full(n, np.nan, dtype=float)
    jerk = np.full(n, np.nan, dtype=float)

    if n < 2:
        return {
            "frames": frames,
            "disp": disp,
            "speed": speed,
            "acc": acc,
            "jerk": jerk,
        }

    for i in range(1, n):
        if not valid[i] or not valid[i - 1]:
            continue
        if np.any(~np.isfinite(coords[i])) or np.any(~np.isfinite(coords[i - 1])):
            continue
        frame_gap = max(1, int(frames[i] - frames[i - 1]))
        d = float(np.linalg.norm(coords[i] - coords[i - 1])) / float(frame_gap)
        disp[i] = d
        speed[i] = d * float(fps)

    for i in range(2, n):
        if np.isfinite(speed[i]) and np.isfinite(speed[i - 1]):
            acc[i] = abs(speed[i] - speed[i - 1]) * float(fps)

    for i in range(3, n):
        if np.isfinite(acc[i]) and np.isfinite(acc[i - 1]):
            jerk[i] = abs(acc[i] - acc[i - 1]) * float(fps)

    return {
        "frames": frames,
        "disp": disp,
        "speed": speed,
        "acc": acc,
        "jerk": jerk,
    }


# ===============================================
# event segment
# ================================================
def mask_to_segments(frames, mask, min_len):
    segs = []
    n = len(frames)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue

        s = i
        e = i
        i += 1
        while i < n and mask[i] and (frames[i] - frames[i - 1] <= 1):
            e = i
            i += 1

        if e - s + 1 >= int(min_len):
            segs.append((s, e))

    return segs


def build_event_row(case_id, method, frames, s_idx, e_idx, fps, event_type, severity_score, evidence_dict):
    start_frame = int(frames[s_idx])
    end_frame = int(frames[e_idx])

    return {
        "case_id": normalize_case_id(case_id),
        "method": method,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_s": round(start_frame / float(fps), 4),
        "end_time_s": round(end_frame / float(fps), 4),
        "event_type": event_type,
        "severity": round(float(severity_score), 4) if severity_score is not None and np.isfinite(severity_score) else 0.0,
        "evidence": evidence_dict,
    }


def detect_events(track, case_id, method, fps=DEFAULT_FPS, k_mad=DEFAULT_K_MAD, min_len=DEFAULT_MIN_LEN):
    frames = track["frames"]
    valid = track["valid"]
    conf = track["conf"]

    metrics = compute_track_metrics(track, fps=fps)
    disp = metrics["disp"]
    jerk = metrics["jerk"]

    events = []

    # jump outlier
    jump_thr, jump_st = robust_high_threshold(disp[np.isfinite(disp)], k_mad)
    jump_guard = np.nan
    if np.isfinite(jump_st.get("p95", np.nan)):
        # guard by tail level to avoid over-sensitive trigger on very smooth tracks
        jump_guard = float(jump_st["p95"]) * 1.2
    if np.isfinite(jump_thr) and np.isfinite(jump_guard):
        jump_thr = max(float(jump_thr), float(jump_guard))
    jump_mask = np.isfinite(disp) & np.isfinite(jump_thr) & (disp > jump_thr)
    for s, e in mask_to_segments(frames, jump_mask, min_len=min_len):
        seg_vals = disp[s : e + 1]
        seg_max = float(np.nanmax(seg_vals))
        sev = (seg_max - jump_thr) / (jump_st["sigma"] + 1e-9)
        ev = build_event_row(
            case_id,
            method,
            frames,
            s,
            e,
            fps,
            "JUMP_OUTLIER",
            sev,
            {
                "threshold": {"type": "median+k*MAD", "k": float(k_mad), "value": float(jump_thr)},
                "stats": {
                    "max_disp": round(seg_max, 6),
                    "p95_disp": round(float(np.nanpercentile(seg_vals, 95)), 6),
                    "disp_median": round(float(jump_st["median"]), 6),
                    "disp_mad": round(float(jump_st["mad"]), 6),
                },
                "severity_level": severity_level(sev),
            },
        )
        events.append(ev)

    # jitter spike
    jitter_thr, jitter_st = robust_high_threshold(jerk[np.isfinite(jerk)], k_mad)
    jitter_guard = np.nan
    if np.isfinite(jitter_st.get("p95", np.nan)):
        jitter_guard = float(jitter_st["p95"]) * 1.2
    if np.isfinite(jitter_thr) and np.isfinite(jitter_guard):
        jitter_thr = max(float(jitter_thr), float(jitter_guard))
    jitter_mask = np.isfinite(jerk) & np.isfinite(jitter_thr) & (jerk > jitter_thr)
    for s, e in mask_to_segments(frames, jitter_mask, min_len=min_len):
        seg_vals = jerk[s : e + 1]
        seg_max = float(np.nanmax(seg_vals))
        sev = (seg_max - jitter_thr) / (jitter_st["sigma"] + 1e-9)
        ev = build_event_row(
            case_id,
            method,
            frames,
            s,
            e,
            fps,
            "JITTER_SPIKE",
            sev,
            {
                "threshold": {"type": "median+k*MAD", "k": float(k_mad), "value": float(jitter_thr)},
                "stats": {
                    "max_jerk": round(seg_max, 6),
                    "jerk_rms": round(float(np.sqrt(np.nanmean(seg_vals * seg_vals))), 6),
                    "jerk_median": round(float(jitter_st["median"]), 6),
                    "jerk_mad": round(float(jitter_st["mad"]), 6),
                },
                "severity_level": severity_level(sev),
            },
        )
        events.append(ev)

    # stall stop
    stall_thr, stall_st = robust_low_threshold(disp[np.isfinite(disp)], k_mad)
    if np.isfinite(stall_thr):
        stall_thr = max(0.0, float(stall_thr))
    stall_gate = np.nan
    if np.isfinite(stall_st.get("median", np.nan)):
        stall_gate = 0.5 * float(stall_st["median"])
    stall_mask = np.isfinite(disp) & np.isfinite(stall_thr) & valid & (disp < stall_thr)
    if np.isfinite(stall_gate):
        stall_mask = stall_mask & (disp < stall_gate)

    # stall should be sustained, use longer min length
    stall_min_len = max(int(min_len), int(max(5, round(float(fps) * 0.5))))

    for s, e in mask_to_segments(frames, stall_mask, min_len=stall_min_len):
        seg_vals = disp[s : e + 1]
        seg_min = float(np.nanmin(seg_vals))
        sev = (stall_thr - seg_min) / (stall_st["sigma"] + 1e-9)
        ev = build_event_row(
            case_id,
            method,
            frames,
            s,
            e,
            fps,
            "STALL_STOP",
            sev,
            {
                "threshold": {"type": "median-k*MAD", "k": float(k_mad), "value": float(stall_thr)},
                "stats": {
                    "min_disp": round(seg_min, 6),
                    "p05_disp": round(float(np.nanpercentile(seg_vals, 5)), 6),
                    "disp_median": round(float(stall_st["median"]), 6),
                    "disp_mad": round(float(stall_st["mad"]), 6),
                },
                "severity_level": severity_level(sev),
            },
        )
        events.append(ev)

    # missing / low valid
    # default use strict missing/invalid only, avoid over-tag on confidence tail
    conf_fin = conf[np.isfinite(conf)]
    conf_low_thr = np.nan
    conf_st = None
    if conf_fin.size > 0:
        conf_low_thr, conf_st = robust_low_threshold(conf_fin, k_mad)

    miss_mask = (~valid)
    for s, e in mask_to_segments(frames, miss_mask, min_len=min_len):
        seg_valid = valid[s : e + 1]
        miss_ratio = 1.0 - float(np.mean(seg_valid.astype(float)))
        sev = miss_ratio * 10.0

        ev_detail = {
            "threshold": {"type": "valid/low-conf", "k": float(k_mad), "conf_low": None if not np.isfinite(conf_low_thr) else float(conf_low_thr)},
            "stats": {
                "missing_ratio": round(miss_ratio, 6),
                "missing_frames": int(np.sum(~seg_valid)),
                "segment_len": int(e - s + 1),
            },
            "severity_level": severity_level(sev),
        }

        if conf_st is not None:
            ev_detail["stats"]["conf_median"] = round(float(conf_st["median"]), 6)
            ev_detail["stats"]["conf_mad"] = round(float(conf_st["mad"]), 6)

        ev = build_event_row(
            case_id,
            method,
            frames,
            s,
            e,
            fps,
            "MISSING_LOW_VALID",
            sev,
            ev_detail,
        )
        events.append(ev)

    events.sort(key=lambda r: (int(r["start_frame"]), int(r["end_frame"]), str(r["event_type"])))

    return events, {
        "frames": frames,
        "disp": disp,
        "speed": metrics["speed"],
        "acc": metrics["acc"],
        "jerk": jerk,
        "jump_thr": jump_thr,
        "jitter_thr": jitter_thr,
        "stall_thr": stall_thr,
        "conf_low_thr": conf_low_thr,
        "valid_ratio": float(np.mean(valid.astype(float))) if len(valid) > 0 else 0.0,
    }


# ===============================================
# public api
# ================================================
def detect_case_events(
    case_id,
    method,
    fps=DEFAULT_FPS,
    k_mad=DEFAULT_K_MAD,
    min_len=DEFAULT_MIN_LEN,
    kpt_id="auto",
    csv_path=None,
    view_side=DEFAULT_VIEW_SIDE,
    continuity=True,
    k_gate=6.0,
    track_window=30,
):
    method = str(method).strip().lower()
    if method not in ("2d", "tri", "monster"):
        raise ValueError(f"unsupported method: {method}")

    case_id_norm = normalize_case_id(case_id)
    if csv_path is None:
        csv_path = DEFAULT_METHOD_CSV[method]

    # 1) load rows
    headers, rows = load_case_rows(csv_path, case_id=case_id_norm, view_side=view_side)
    if len(rows) == 0:
        return {
            "case_id": case_id_norm,
            "method": method,
            "csv_path": str(csv_path),
            "kpt_id": "",
            "view_side": str(view_side),
            "track": {
                "frames": np.asarray([], dtype=int),
                "coords": np.asarray([], dtype=float),
                "valid": np.asarray([], dtype=bool),
                "conf": np.asarray([], dtype=float),
                "switch_flag": np.asarray([], dtype=bool),
                "dim": 2,
            },
            "events": [],
            "metrics": {
                "frames": np.asarray([], dtype=int),
                "disp": np.asarray([], dtype=float),
                "speed": np.asarray([], dtype=float),
                "acc": np.asarray([], dtype=float),
                "jerk": np.asarray([], dtype=float),
                "valid_ratio": 0.0,
            },
        }

    # 2) parse schema + kpt
    schema = infer_schema(headers, method=method)
    spec = choose_kpt_spec(schema, rows, forced_kpt=kpt_id)
    if spec is None:
        raise RuntimeError(f"cannot determine kpt columns for method={method}, csv={csv_path}")

    # 3) build track
    track = build_track(
        rows,
        spec,
        continuity=bool(continuity),
        k_gate=float(k_gate),
        window=int(track_window),
    )

    # 4) detect events
    events, metrics = detect_events(
        track,
        case_id=case_id_norm,
        method=method,
        fps=fps,
        k_mad=k_mad,
        min_len=min_len,
    )

    kpt_name = spec.get("kpt_name", spec.get("kpt_value", ""))

    return {
        "case_id": case_id_norm,
        "method": method,
        "csv_path": str(csv_path),
        "kpt_id": str(kpt_name),
        "view_side": str(view_side),
        "track": track,
        "events": events,
        "metrics": metrics,
    }


# ===============================================
# io
# ================================================
def save_events_csv(events, out_csv_path):
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "case_id",
        "method",
        "start_frame",
        "end_frame",
        "start_time_s",
        "end_time_s",
        "event_type",
        "severity",
        "evidence",
    ]

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in events:
            rr = dict(r)
            ev = rr.get("evidence", {})
            rr["evidence"] = json.dumps(ev, ensure_ascii=False)
            writer.writerow(rr)
