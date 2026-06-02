"""
Evaluate kalman result by case and method.
Output: evaluate_kalman.csv (case_id + method)
"""
import csv
import math
from pathlib import Path

import numpy as np


# ===============================================
# config
# ================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
DET_DIR = REPO_ROOT / "output" / "detections"
EVAL_DIR = REPO_ROOT / "output" / "evaluation"

ROUND_DIGITS = 6
KPTS = ["L1", "L2", "R1", "R2"]
KPT_2D_GROUPS = {k: [f"{k}_x", f"{k}_y"] for k in KPTS}
KPT_3D_GROUPS = {k: [f"{k}_x3d", f"{k}_y3d", f"{k}_z3d"] for k in KPTS}
POLICY_CSV_PATH = EVAL_DIR / "policy.csv"


# safe float
# ================================================
def safe_float(x, default=None):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


# safe int
# ================================================
def safe_int(x, default=None):
    try:
        if x is None or x == "":
            return default
        return int(x)
    except (TypeError, ValueError):
        return default


# parse filename
# ================================================
def parse_filename(filename):
    stem = Path(str(filename)).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    if not parts[0].isdigit() or not parts[-1].isdigit():
        return None

    case_id = int(parts[0])
    frame_idx = int(parts[-1])
    name = "_".join(parts[1:-1]).lower()
    side = "left" if "left" in name else "right" if "right" in name else "unknown"
    return case_id, side, frame_idx


# normalize mode
# ================================================
def normalize_mode(mode):
    m = str(mode or "").strip().lower()
    if m in ("2d", "tri", "monster"):
        return m
    return "None"


# load policy mode
# ================================================
def load_policy_mode_by_case(policy_csv_path):
    policy_csv_path = Path(policy_csv_path)
    data = {}
    if not policy_csv_path.exists():
        return data

    with open(policy_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = safe_int(row.get("case_id"), default=None)
            if cid is None:
                continue
            data[cid] = normalize_mode(row.get("mode", ""))
    return data


# distance
# ================================================
def vec_dist(a, b):
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


# jitter
# ================================================
def vec_jitter(a, b, c):
    return math.sqrt(sum((x3 - 2.0 * x2 + x1) ** 2 for x1, x2, x3 in zip(a, b, c)))


# median
# ================================================
def med(vals):
    return float(np.nanmedian(vals)) if vals else None


# p95
# ================================================
def p95(vals):
    return float(np.nanpercentile(vals, 95)) if vals else None


# fmt
# ================================================
def fmt_num(x):
    if x is None:
        return ""
    return round(float(x), ROUND_DIGITS)


# read vector
# ================================================
def read_vec(row, cols):
    vals = []
    for c in cols:
        v = safe_float(row.get(c), default=None)
        if v is None or not np.isfinite(v):
            return None
        vals.append(float(v))
    return tuple(vals)


# load by case
# ================================================
def load_kalman_by_case(csv_path):
    csv_path = Path(csv_path)
    by_case = {}

    if not csv_path.exists():
        return by_case

    # The kalman csv already contains both raw and *_kf columns.
    # Here we only group rows by case for later per-method comparison.
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)

            cid = safe_int(row.get("case_id"), default=None)
            frame_idx = safe_int(row.get("frame_idx"), default=None)
            side = str(row.get("side", "")).strip().lower()

            if cid is None or frame_idx is None or side == "":
                parsed = parse_filename(row.get("filename", ""))
                if parsed is None:
                    continue
                cid, side_p, frame_p = parsed
                if frame_idx is None:
                    frame_idx = frame_p
                if side == "":
                    side = side_p

            by_case.setdefault(cid, [])
            by_case[cid].append({"row": row, "side": side, "frame_idx": frame_idx})

    return by_case


# compute one case metrics
# ================================================
def compute_case_metrics(entries, groups):
    disp_raw = []
    disp_kf = []
    jitter_raw = []
    jitter_kf = []

    entries = list(entries)
    entries.sort(key=lambda x: (x["side"], x["frame_idx"]))

    # Evaluate each camera side separately, then pool the distance and jitter
    # values to get one case-level summary for the chosen method.
    for side in ["left", "right", "unknown"]:
        side_entries = [e for e in entries if e["side"] == side]
        if not side_entries:
            continue

        for k in KPTS:
            cols_raw = groups[k]
            cols_kf = [f"{c}_kf" for c in cols_raw]

            seq_raw = []
            seq_kf = []
            for e in side_entries:
                v_raw = read_vec(e["row"], cols_raw)
                v_kf = read_vec(e["row"], cols_kf)

                if v_raw is not None:
                    seq_raw.append(v_raw)
                if v_kf is not None:
                    seq_kf.append(v_kf)

            # disp_p95 reflects frame-to-frame motion amplitude.
            for i in range(1, len(seq_raw)):
                disp_raw.append(vec_dist(seq_raw[i - 1], seq_raw[i]))
            for i in range(1, len(seq_kf)):
                disp_kf.append(vec_dist(seq_kf[i - 1], seq_kf[i]))

            # jitter_med uses a second-order difference and is more sensitive
            # to high-frequency wobble than plain displacement.
            for i in range(2, len(seq_raw)):
                jitter_raw.append(vec_jitter(seq_raw[i - 2], seq_raw[i - 1], seq_raw[i]))
            for i in range(2, len(seq_kf)):
                jitter_kf.append(vec_jitter(seq_kf[i - 2], seq_kf[i - 1], seq_kf[i]))

    disp_p95_raw = p95(disp_raw)
    disp_p95_kf = p95(disp_kf)
    jitter_med_raw = med(jitter_raw)
    jitter_med_kf = med(jitter_kf)

    improve_disp = (disp_p95_raw - disp_p95_kf) if (disp_p95_raw is not None and disp_p95_kf is not None) else None
    improve_jitter = (jitter_med_raw - jitter_med_kf) if (jitter_med_raw is not None and jitter_med_kf is not None) else None

    return {
        "disp_p95_raw": disp_p95_raw,
        "disp_p95_kf": disp_p95_kf,
        "improve_disp_p95": improve_disp,
        "jitter_med_raw": jitter_med_raw,
        "jitter_med_kf": jitter_med_kf,
        "improve_jitter_med": improve_jitter,
    }


# evaluate one method
# ================================================
def evaluate_method(csv_path, method_name, groups, allowed_cases=None):
    by_case = load_kalman_by_case(csv_path)
    rows = []

    for cid in sorted(by_case.keys()):
        if allowed_cases is not None and cid not in allowed_cases:
            continue
        m = compute_case_metrics(by_case[cid], groups)
        rows.append({
            "case_id": cid,
            "method": method_name,
            **m,
        })

    return rows


# save csv
# ================================================
def save_csv(rows, out_csv_path):
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "case_id",
        "method",
        "disp_p95_raw",
        "disp_p95_kf",
        "improve_disp_p95",
        "jitter_med_raw",
        "jitter_med_kf",
        "improve_jitter_med",
    ]

    rows = list(rows)
    rows.sort(key=lambda r: (safe_int(r.get("case_id"), 0), str(r.get("method", ""))))

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            for c in cols:
                if c in ("case_id", "method"):
                    continue
                rr[c] = fmt_num(rr.get(c))
            w.writerow(rr)

    print(f"[DONE] saved: {out_csv_path}")


# main
# ================================================
def main():
    det_dir = DET_DIR
    eval_dir = EVAL_DIR

    policy_mode_by_case = load_policy_mode_by_case(POLICY_CSV_PATH)
    allow_2d = None
    allow_tri = None
    allow_mon = None
    if policy_mode_by_case:
        allow_2d = {cid for cid, mode in policy_mode_by_case.items() if mode in ("2d", "tri", "monster")}
        allow_tri = {cid for cid, mode in policy_mode_by_case.items() if mode in ("tri", "monster")}
        allow_mon = {cid for cid, mode in policy_mode_by_case.items() if mode == "monster"}

    rows = []

    jobs = [
        (det_dir / "2d_kalman_results.csv", "2d", KPT_2D_GROUPS, allow_2d),
        (det_dir / "3d_tri_kalman_results.csv", "3d_tri", KPT_3D_GROUPS, allow_tri),
        (det_dir / "3d_monster_kalman_results.csv", "3d_monster", KPT_3D_GROUPS, allow_mon),
    ]

    for csv_path, method_name, groups, allowed_cases in jobs:
        rows.extend(evaluate_method(csv_path, method_name, groups, allowed_cases=allowed_cases))

    save_csv(rows, eval_dir / "evaluate_kalman.csv")


if __name__ == "__main__":
    main()
