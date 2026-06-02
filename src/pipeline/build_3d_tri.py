"""
Step B1: build tri 3D from 2d_results.csv and stereo ini.
"""
import csv
import re
import configparser
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# ===============================================
# config
# ================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
DET_DIR = REPO_ROOT / "output" / "detections"
DATASET_DIR = REPO_ROOT / "data" / "dataset"

IN_CSV_PATH = DET_DIR / "2d_results.csv"
CALIB_ROOT = DATASET_DIR
OUT_CSV_PATH = DET_DIR / "3d_tri_results.csv"
EVAL_DIR = REPO_ROOT / "output" / "evaluation"
EVAL_2D_RESULTS_PATH = EVAL_DIR / "evaluate_2d_results.csv"

KPT_LABELS = ["L1", "L2", "R1", "R2"]

# These gates are part of the current TRI logic.
# They are not only for reporting. They decide whether one stereo pair
# is stable enough to become a usable 3D observation.
TRI_MAX_Y_DIFF = 220.0
TRI_MIN_DISP = 0.5
TRI_MAX_Z = 1200.0
TRI_MAX_REPROJ_ERR = 80.0
TRI_MIN_CONF = 0.20
TRI_MIN_RAY_ANGLE_DEG = 0.20
TRI_POST_RANGE_Q_LOW = 0.5
TRI_POST_RANGE_Q_HIGH = 99.5
TRI_POST_RANGE_PAD = 0.20
TRI_POST_JUMP_K = 8.0
TRI_POST_SMOOTH_WIN = 3
TRI_POST_Z_MIN = 20.0
TRI_POST_Z_MAX = 295.0

try:
    from src.pipeline.policy import GatingPolicy
except Exception:
    GatingPolicy = None


# parse filename
# ================================================
def parse_filename(filename):
    stem = Path(str(filename)).stem
    m = re.match(r"^(\d+)_(left|right)_video_(\d+)$", stem, re.I)
    if m is None:
        return None, None, None
    return m.group(1).zfill(6), m.group(2).lower(), int(m.group(3))


# safe float
# ================================================
def safe_float(x, default=np.nan):
    if x is None or x == "":
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# get confidence
# ================================================
def get_kpt_conf(row, label):
    conf = safe_float(row.get(f"{label}_conf"), default=np.nan)
    if np.isfinite(conf):
        return conf

    side_conf_name = "L_conf" if str(label).startswith("L") else "R_conf"
    conf2 = safe_float(row.get(side_conf_name), default=np.nan)
    return conf2


# case id file
# ================================================
def load_case_filter(case_list_path):
    if case_list_path is None:
        return None

    case_list_path = Path(case_list_path)
    if not case_list_path.exists():
        print(f"[WARNING] case list not found: {case_list_path}")
        return None

    case_ids = set()
    with open(case_list_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text == "" or text.lower() == "case_id":
                continue
            part = text.split(",")[0].strip()
            if part.isdigit():
                case_ids.add(part.zfill(6))

    print(f"[INFO] case filter loaded: {len(case_ids)}")
    return case_ids


# eval csv by case
# ================================================
def load_eval_by_case(csv_path):
    csv_path = Path(csv_path)
    data = {}
    if not csv_path.exists():
        return data

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = str(row.get("case_id", "")).strip()
            if case_id == "" or case_id.upper() == "ALL":
                continue
            if not case_id.isdigit():
                continue
            data[case_id.zfill(6)] = row
    return data


# build filter per case, from 2d eval and policy
# ================================================
def build_case_filter_from_eval(eval_2d_path):
    if GatingPolicy is None:
        print("[WARNING] policy module import failed, skip 2d gate filter")
        return None

    eval_2d_path = Path(eval_2d_path)
    if not eval_2d_path.exists():
        print(f"[WARNING] eval file not found: {eval_2d_path}")
        return None

    policy_obj = GatingPolicy()
    eval_2d = load_eval_by_case(eval_2d_path)
    case_ids = set()
    for cid, row in eval_2d.items():
        if policy_obj.evaluate_2d(row):
            case_ids.add(cid)

    # TRI is only attempted on cases that already passed the 2D gate.
    # This keeps impossible or low-quality cases out of the 3D stage.
    print(f"[INFO] tri target cases from policy 2d gate: {len(case_ids)}")
    return case_ids


# load calib
# ================================================
def load_stereo_calib(ini_path):
    cfg = configparser.ConfigParser()
    cfg.read(ini_path, encoding="utf-8")

    def read_k(section):
        fx = float(cfg.get(section, "fc_x"))
        fy = float(cfg.get(section, "fc_y"))
        cx = float(cfg.get(section, "cc_x"))
        cy = float(cfg.get(section, "cc_y"))
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    k_left = read_k("StereoLeft")
    k_right = read_k("StereoRight")

    def read_dist(section):
        vals = []
        for i in range(5):
            vals.append(float(cfg.get(section, f"kc_{i}", fallback="0")))
        return np.array(vals, dtype=np.float64)

    d_left = read_dist("StereoLeft")
    d_right = read_dist("StereoRight")

    r = np.array([
        [float(cfg.get("StereoRight", "R_0")), float(cfg.get("StereoRight", "R_1")), float(cfg.get("StereoRight", "R_2"))],
        [float(cfg.get("StereoRight", "R_3")), float(cfg.get("StereoRight", "R_4")), float(cfg.get("StereoRight", "R_5"))],
        [float(cfg.get("StereoRight", "R_6")), float(cfg.get("StereoRight", "R_7")), float(cfg.get("StereoRight", "R_8"))],
    ], dtype=np.float64)

    t = np.array([
        float(cfg.get("StereoRight", "T_0")),
        float(cfg.get("StereoRight", "T_1")),
        float(cfg.get("StereoRight", "T_2")),
    ], dtype=np.float64)

    return k_left, d_left, k_right, d_right, r, t


# calib path
# ================================================
def find_calib_ini_path(calib_root, case_id):
    calib_root = Path(calib_root)
    p1 = calib_root / case_id / "StereoCalibrationDVRK.ini"
    if p1.exists():
        return p1
    p2 = calib_root / case_id / case_id / "StereoCalibrationDVRK.ini"
    if p2.exists():
        return p2
    # Some local bundles only keep one shared stereo calibration file.
    # When case folders are not shipped, fall back to 000000 so TRI can
    # still be reproduced from the packed project files.
    p3 = calib_root / "000000" / "StereoCalibrationDVRK.ini"
    if p3.exists():
        return p3
    return None


# projection
# ================================================
def build_projection_matrices(k_left, k_right, r, t):
    p1 = k_left @ np.hstack([np.eye(3), np.zeros((3, 1))])
    p2 = k_right @ np.hstack([r, t.reshape(3, 1)])
    return p1, p2


# point tri
# ================================================
def triangulate_point(p1, p2, xl, yl, xr, yr):
    pt_l = np.array([[xl, yl]], dtype=np.float32).T
    pt_r = np.array([[xr, yr]], dtype=np.float32).T

    x4 = cv2.triangulatePoints(p1, p2, pt_l, pt_r).ravel()
    if x4[3] == 0:
        return np.nan, np.nan, np.nan

    return float(x4[0] / x4[3]), float(x4[1] / x4[3]), float(x4[2] / x4[3])


# undistort one point
# ================================================
def undistort_xy(x, y, k_mat, d_vec):
    pt = np.array([[[float(x), float(y)]]], dtype=np.float64)
    und = cv2.undistortPoints(pt, k_mat, d_vec, P=k_mat)
    return float(und[0, 0, 0]), float(und[0, 0, 1])


# project point
# ================================================
def project_point(p, x, y, z):
    xh = p @ np.array([x, y, z, 1.0], dtype=np.float64)
    if abs(float(xh[2])) < 1e-9:
        return np.nan, np.nan
    return float(xh[0] / xh[2]), float(xh[1] / xh[2])


# ray angle
# ================================================
def ray_angle_deg(xl, yl, xr, yr, k_left, k_right, r):
    inv_l = np.linalg.inv(k_left)
    inv_r = np.linalg.inv(k_right)

    ray_l = inv_l @ np.array([float(xl), float(yl), 1.0], dtype=np.float64)
    ray_r = inv_r @ np.array([float(xr), float(yr), 1.0], dtype=np.float64)

    n_l = float(np.linalg.norm(ray_l))
    n_r = float(np.linalg.norm(ray_r))
    if n_l < 1e-9 or n_r < 1e-9:
        return 0.0

    ray_l = ray_l / n_l
    ray_r = ray_r / n_r

    # right camera ray -> left camera frame
    ray_r_in_left = r.T @ ray_r
    n_rl = float(np.linalg.norm(ray_r_in_left))
    if n_rl < 1e-9:
        return 0.0
    ray_r_in_left = ray_r_in_left / n_rl

    c = float(np.dot(ray_l, ray_r_in_left))
    c = min(1.0, max(-1.0, c))
    return float(np.degrees(np.arccos(c)))


# load rows
# ================================================
def load_rows(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[ERROR] csv not found: {csv_path}")
        return [], []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        raw_rows = [dict(r) for r in reader]

    rows = []
    skipped = 0
    for row in raw_rows:
        case_id, side, frame_idx = parse_filename(row.get("filename", ""))
        if case_id is None:
            skipped += 1
            continue
        rows.append(row)

    print(f"[INFO] loaded rows={len(rows)}, skipped_bad_name={skipped}")
    return fieldnames, rows


# tri rows
# ================================================
def triangulate_rows(rows, calib_root, case_filter=None):
    pair_rows = defaultdict(dict)
    for row in rows:
        case_id, side, frame_idx = parse_filename(row.get("filename", ""))
        if case_id is None:
            continue
        pair_rows[(case_id, frame_idx)][side] = row

    extra_cols = [f"{k}_{a}3d" for k in KPT_LABELS for a in ["x", "y", "z"]]
    for row in rows:
        for c in extra_cols:
            row[c] = ""

    calib_cache = {}
    geo_stats = {
        "candidates": 0,
        "skip_conf": 0,
        "skip_y_diff": 0,
        "skip_disp": 0,
        "skip_ray_angle": 0,
        "skip_depth": 0,
        "skip_cheirality": 0,
        "skip_reproj": 0,
        "pass": 0,
        "reproj_err": [],
        "ray_angle_deg": [],
    }

    # Process stereo pairs in a stable order so csv output is reproducible.
    pair_keys = sorted(pair_rows.keys(), key=lambda x: (x[0], x[1]))
    for case_id, frame_idx in pair_keys:
        side_rows = pair_rows[(case_id, frame_idx)]
        if "left" not in side_rows or "right" not in side_rows:
            continue
        if case_filter is not None and case_id not in case_filter:
            continue

        left_row = side_rows["left"]
        right_row = side_rows["right"]

        if case_id not in calib_cache:
            ini_path = find_calib_ini_path(calib_root, case_id)
            if ini_path is None:
                continue
            k_left, d_left, k_right, d_right, r, t = load_stereo_calib(str(ini_path))
            p1, p2 = build_projection_matrices(k_left, k_right, r, t)
            calib_cache[case_id] = (p1, p2, k_left, d_left, k_right, d_right, r, t)

        p1, p2, k_left, d_left, k_right, d_right, r, t = calib_cache[case_id]

        for label in KPT_LABELS:
            xl = safe_float(left_row.get(f"{label}_x"))
            yl = safe_float(left_row.get(f"{label}_y"))
            xr = safe_float(right_row.get(f"{label}_x"))
            yr = safe_float(right_row.get(f"{label}_y"))
            cl = get_kpt_conf(left_row, label)
            cr = get_kpt_conf(right_row, label)

            if np.isnan(xl) or np.isnan(yl) or np.isnan(xr) or np.isnan(yr):
                continue

            geo_stats["candidates"] += 1

            # === 1. stereo gate
            # We first reject obviously weak stereo pairs:
            # low confidence, broken vertical alignment, or near-zero disparity.
            if np.isfinite(cl) and cl < TRI_MIN_CONF:
                geo_stats["skip_conf"] += 1
                continue
            if np.isfinite(cr) and cr < TRI_MIN_CONF:
                geo_stats["skip_conf"] += 1
                continue
            xl_u, yl_u = undistort_xy(xl, yl, k_left, d_left)
            xr_u, yr_u = undistort_xy(xr, yr, k_right, d_right)
            if abs(yl_u - yr_u) > TRI_MAX_Y_DIFF:
                geo_stats["skip_y_diff"] += 1
                continue
            if abs(xl_u - xr_u) < TRI_MIN_DISP:
                geo_stats["skip_disp"] += 1
                continue

            # === 2. ray angle gate
            # Tiny ray angle means the two views are almost parallel.
            # In that case depth becomes numerically unstable.
            ang = ray_angle_deg(xl_u, yl_u, xr_u, yr_u, k_left, k_right, r)
            if ang < TRI_MIN_RAY_ANGLE_DEG:
                geo_stats["skip_ray_angle"] += 1
                continue

            x3, y3, z3 = triangulate_point(p1, p2, xl_u, yl_u, xr_u, yr_u)

            # === 3. depth gate
            # Keep only finite points in front of the camera and inside
            # a reasonable depth range for this project.
            if np.isnan(x3) or np.isnan(y3) or np.isnan(z3):
                geo_stats["skip_depth"] += 1
                continue
            if z3 <= 0.0 or z3 > TRI_MAX_Z:
                geo_stats["skip_depth"] += 1
                continue

            # The reconstructed point must also stay in front of the right camera.
            xr_cam = (r @ np.array([x3, y3, z3], dtype=np.float64)) + t.reshape(3)
            if float(xr_cam[2]) <= 0.0:
                geo_stats["skip_cheirality"] += 1
                continue

            # === 4. reprojection gate
            # Reproject the 3D point back to both images. If the pixel error
            # is too large, the triangulated point is not trusted.
            ul, vl = project_point(p1, x3, y3, z3)
            ur, vr = project_point(p2, x3, y3, z3)
            if np.isnan(ul) or np.isnan(vl) or np.isnan(ur) or np.isnan(vr):
                geo_stats["skip_reproj"] += 1
                continue
            err_l = float(np.sqrt((ul - xl_u) ** 2 + (vl - yl_u) ** 2))
            err_r = float(np.sqrt((ur - xr_u) ** 2 + (vr - yr_u) ** 2))
            if err_l > TRI_MAX_REPROJ_ERR or err_r > TRI_MAX_REPROJ_ERR:
                geo_stats["skip_reproj"] += 1
                continue

            geo_stats["pass"] += 1
            geo_stats["ray_angle_deg"].append(float(ang))
            geo_stats["reproj_err"].append(float(max(err_l, err_r)))

            val_x = "" if np.isnan(x3) else round(x3, 4)
            val_y = "" if np.isnan(y3) else round(y3, 4)
            val_z = "" if np.isnan(z3) else round(z3, 4)

            for row in (left_row, right_row):
                row[f"{label}_x3d"] = val_x
                row[f"{label}_y3d"] = val_y
                row[f"{label}_z3d"] = val_z

    # === 5. post process
    # After per-frame geometry checks, we still do a sequence clean-up step.
    # This removes isolated spikes and smooths valid trajectories a little.
    postprocess_triangulated_rows(pair_rows)

    return rows, extra_cols, geo_stats


# print geo summary
# ================================================
def print_geo_stats(geo_stats):
    if geo_stats is None:
        return

    cands = int(geo_stats.get("candidates", 0))
    passed = int(geo_stats.get("pass", 0))
    ratio = 0.0 if cands <= 0 else float(passed) / float(cands)

    print(
        "[INFO] tri geo check: "
        f"candidates={cands}, pass={passed}, pass_ratio={ratio:.4f}, "
        f"skip_conf={int(geo_stats.get('skip_conf', 0))}, "
        f"skip_y={int(geo_stats.get('skip_y_diff', 0))}, "
        f"skip_disp={int(geo_stats.get('skip_disp', 0))}, "
        f"skip_angle={int(geo_stats.get('skip_ray_angle', 0))}, "
        f"skip_depth={int(geo_stats.get('skip_depth', 0))}, "
        f"skip_cheirality={int(geo_stats.get('skip_cheirality', 0))}, "
        f"skip_reproj={int(geo_stats.get('skip_reproj', 0))}"
    )

    errs = np.asarray(geo_stats.get("reproj_err", []), dtype=float)
    errs = errs[np.isfinite(errs)]
    if errs.size > 0:
        print(
            "[INFO] reproj max(err_l,err_r): "
            f"p50={float(np.percentile(errs, 50)):.3f}, "
            f"p95={float(np.percentile(errs, 95)):.3f}, "
            f"max={float(np.max(errs)):.3f}"
        )

    ang = np.asarray(geo_stats.get("ray_angle_deg", []), dtype=float)
    ang = ang[np.isfinite(ang)]
    if ang.size > 0:
        print(
            "[INFO] ray angle deg: "
            f"p50={float(np.percentile(ang, 50)):.3f}, "
            f"p95={float(np.percentile(ang, 95)):.3f}, "
            f"min={float(np.min(ang)):.3f}"
        )


# robust jump threshold
# ================================================
def robust_jump_threshold(vals, k=6.0):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 8:
        return np.inf

    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = float(1.4826 * mad)
    if sigma < 1e-9:
        sigma = float(np.std(arr)) + 1e-9
    return med + float(k) * sigma


# postprocess tri rows
# ================================================
def postprocess_triangulated_rows(pair_rows):
    keys_sorted = sorted(pair_rows.keys(), key=lambda x: (x[0], x[1]))

    for label in KPT_LABELS:
        by_case = defaultdict(list)
        for case_id, frame_idx in keys_sorted:
            sides = pair_rows[(case_id, frame_idx)]
            if "left" not in sides or "right" not in sides:
                continue

            row_left = sides["left"]
            row_right = sides["right"]
            x = safe_float(row_left.get(f"{label}_x3d"))
            y = safe_float(row_left.get(f"{label}_y3d"))
            z = safe_float(row_left.get(f"{label}_z3d"))
            by_case[case_id].append([frame_idx, row_left, row_right, x, y, z])

        for case_id, seq in by_case.items():
            if len(seq) < 10:
                continue
            seq.sort(key=lambda t: t[0])

            frames = np.asarray([int(t[0]) for t in seq], dtype=int)
            arr = np.asarray([[t[3], t[4], t[5]] for t in seq], dtype=float)
            valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2])
            if int(np.sum(valid)) < 8:
                continue

            out = arr.copy()

            # 1) range clamp
            for d in range(3):
                vals = out[valid, d]
                if vals.size < 8:
                    continue
                lo = float(np.percentile(vals, TRI_POST_RANGE_Q_LOW))
                hi = float(np.percentile(vals, TRI_POST_RANGE_Q_HIGH))
                pad = (hi - lo) * float(TRI_POST_RANGE_PAD)
                out[valid, d] = np.clip(out[valid, d], lo - pad, hi + pad)

            # 2) jump replace
            disp = []
            for i in range(1, len(seq)):
                if not (valid[i - 1] and valid[i]):
                    continue
                gap = max(1, int(frames[i] - frames[i - 1]))
                if gap > 2:
                    continue
                d = float(np.linalg.norm(out[i] - out[i - 1]) / float(gap))
                if np.isfinite(d):
                    disp.append(d)

            thr = robust_jump_threshold(disp, k=TRI_POST_JUMP_K)
            if np.isfinite(thr):
                for i in range(1, len(seq)):
                    if not (valid[i - 1] and valid[i]):
                        continue
                    gap = max(1, int(frames[i] - frames[i - 1]))
                    if gap > 2:
                        continue
                    d = float(np.linalg.norm(out[i] - out[i - 1]) / float(gap))
                    if d > thr:
                        out[i] = out[i - 1]

            # 3) small smooth
            win = int(max(3, TRI_POST_SMOOTH_WIN))
            if win % 2 == 0:
                win += 1
            half = win // 2
            for i in range(len(seq)):
                if not valid[i]:
                    continue
                l = max(0, i - half)
                r = min(len(seq), i + half + 1)
                idx = [j for j in range(l, r) if valid[j]]
                if len(idx) >= 2:
                    out[i] = np.mean(out[idx], axis=0)

            # 4) workspace depth clamp
            for i in range(len(seq)):
                if not valid[i]:
                    continue
                out[i, 2] = float(np.clip(out[i, 2], TRI_POST_Z_MIN, TRI_POST_Z_MAX))

            for i, item in enumerate(seq):
                row_left = item[1]
                row_right = item[2]
                if not valid[i]:
                    vx, vy, vz = "", "", ""
                else:
                    vx = round(float(out[i, 0]), 4)
                    vy = round(float(out[i, 1]), 4)
                    vz = round(float(out[i, 2]), 4)
                for row in (row_left, row_right):
                    row[f"{label}_x3d"] = vx
                    row[f"{label}_y3d"] = vy
                    row[f"{label}_z3d"] = vz


# save csv
# ================================================
def save_rows(fieldnames, rows, out_csv, extra_cols):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_fields = list(fieldnames)
    for c in extra_cols:
        if c not in out_fields:
            out_fields.append(c)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] saved: {out_csv}")
    print(f"[INFO] rows: {len(rows)}")


# run core
# ================================================
def run(in_csv_path, calib_root, out_csv_path, case_list_path=None):
    # load csv
    case_filter = load_case_filter(case_list_path)
    if case_filter is None:
        case_filter = build_case_filter_from_eval(EVAL_2D_RESULTS_PATH)
    fieldnames, rows = load_rows(in_csv_path)
    if not rows:
        print("[ERROR] no rows loaded")
        return None

    # triangulate and save
    rows, extra_cols, geo_stats = triangulate_rows(rows, calib_root, case_filter=case_filter)
    save_rows(fieldnames, rows, out_csv_path, extra_cols)
    print_geo_stats(geo_stats)
    return str(out_csv_path)


# main
# ================================================
def main():
    run(IN_CSV_PATH, CALIB_ROOT, OUT_CSV_PATH, case_list_path=None)


if __name__ == "__main__":
    main()
