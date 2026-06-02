import csv
import math
import re
from pathlib import Path

import numpy as np


# ===============================================
# path and settings
# ================================================
REPO_ROOT = Path(__file__).resolve().parents[2]

DET_DIR = REPO_ROOT / "output" / "detections"
EVAL_DIR = REPO_ROOT / "output" / "evaluation"

POLICY_CSV_PATH = EVAL_DIR / "policy.csv"
CSV_2D_PATH = DET_DIR / "2d_results.csv"
CSV_TRI_PATH = DET_DIR / "3d_tri_results.csv"
CSV_MONSTER_PATH = DET_DIR / "3d_monster_results.csv"

OUT_CSV_PATH = DET_DIR / "policy_kalman_results.csv"

ROUND_DIGITS = 4

KALMAN_DT = 1.0
KALMAN_PROCESS_VAR = 1.0
KALMAN_MEASURE_VAR = 25.0
KALMAN_INIT_COV = 50.0

KPT_2D_COLS = ["L1_x", "L1_y", "L2_x", "L2_y", "R1_x", "R1_y", "R2_x", "R2_y"]
KPT_3D_COLS = [
    "L1_x3d", "L1_y3d", "L1_z3d",
    "L2_x3d", "L2_y3d", "L2_z3d",
    "R1_x3d", "R1_y3d", "R1_z3d",
    "R2_x3d", "R2_y3d", "R2_z3d",
]


# ===============================================
# helper
# ===============================================
def safe_float(x, default=None):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


def parse_filename(filename):
    stem = Path(str(filename)).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    if not re.fullmatch(r"\d+", parts[0]):
        return None
    if not parts[-1].isdigit():
        return None

    case_id = int(parts[0])
    frame_idx = int(parts[-1])
    name = "_".join(parts[1:-1]).lower()

    if "left" in name:
        side = "left"
    elif "right" in name:
        side = "right"
    else:
        side = "unknown"

    return case_id, side, frame_idx


def normalize_mode(mode_raw):
    mode_raw = str(mode_raw or "").strip()
    mode = mode_raw.lower()

    if mode == "tri":
        return "tri"
    if mode == "monster":
        return "monster"
    if mode == "2d":
        return "2d"
    if mode in ("none", ""):
        return "None"

    # Keep unknown mode text for summary/diagnosis.
    return mode_raw


def is_bad_mode(mode):
    return mode not in ("tri", "monster", "2d")


def copy_entries(entries):
    out = []
    for e in entries:
        out.append({
            "case_id": e["case_id"],
            "side": e["side"],
            "frame_idx": e["frame_idx"],
            "row": dict(e["row"]),
        })
    return out


def side_rank(side):
    s = str(side).lower()
    if s == "left":
        return 0
    if s == "right":
        return 1
    return 2


# ===============================================
# 0. load files
# ===============================================
def load_policy(policy_path):
    policy_path = Path(policy_path)
    policy_by_case = {}
    raw_mode_by_case = {}

    if not policy_path.exists():
        print(f"[ERROR] {policy_path} does not exist")
        return policy_by_case, raw_mode_by_case

    with open(policy_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_raw = str(row.get("case_id", "")).strip()
            if not case_raw.isdigit():
                continue
            case_id = int(case_raw)
            mode_raw = str(row.get("mode", "")).strip()
            raw_mode_by_case[case_id] = mode_raw
            policy_by_case[case_id] = normalize_mode(mode_raw)

    mode_count = {"tri": 0, "monster": 0, "2d": 0, "None": 0, "other": 0}
    for m in policy_by_case.values():
        if m in mode_count:
            mode_count[m] += 1
        else:
            mode_count["other"] += 1

    print(f"[DONE] Loaded policy: cases={len(policy_by_case)}")
    print(
        f"[INFO] Policy counts: tri={mode_count['tri']}, monster={mode_count['monster']}, "
        f"2d={mode_count['2d']}, None={mode_count['None']}, other={mode_count['other']}"
    )
    return policy_by_case, raw_mode_by_case


def load_detection_by_case(csv_path, tag="source", verbose=True):
    csv_path = Path(csv_path)
    data_by_case = {}
    headers = []

    if not csv_path.exists():
        if verbose:
            print(f"[WARNING] {tag} csv not found: {csv_path}")
        return data_by_case, headers

    total = 0
    skipped = 0

    # Each source csv is indexed by case so policy routing can choose
    # the right trajectory branch later.
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        for row in reader:
            total += 1
            row = dict(row)

            if "filename" not in row:
                for k in list(row.keys()):
                    if "filename" in str(k).lower():
                        row["filename"] = row[k]
                        break

            parsed = parse_filename(row.get("filename", ""))
            if parsed is None:
                skipped += 1
                continue

            case_id, side, frame_idx = parsed
            data_by_case.setdefault(case_id, [])
            data_by_case[case_id].append({
                "case_id": case_id,
                "side": side,
                "frame_idx": frame_idx,
                "row": row,
            })

    for cid in data_by_case:
        data_by_case[cid].sort(key=lambda x: (side_rank(x["side"]), x["frame_idx"]))

    if verbose:
        print(f"[DONE] Loaded {tag}: cases={len(data_by_case)}, rows={sum(len(v) for v in data_by_case.values())}, skipped={skipped}/{total}")
    return data_by_case, headers


# ===============================================
# 2. kalman by policy mode
# ===============================================
def kalman_1d(values):
    dt = KALMAN_DT

    F = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
    H = np.array([[1.0, 0.0]], dtype=float)
    Q = KALMAN_PROCESS_VAR * np.array(
        [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]], dtype=float
    )
    R = np.array([[KALMAN_MEASURE_VAR]], dtype=float)
    I = np.eye(2, dtype=float)

    x = None
    P = None
    outputs = []

    # Run one simple constant-velocity Kalman filter on a single scalar channel.
    # Missing measurements are allowed and only skip the update step.
    for z in values:
        if x is None:
            if z is None:
                outputs.append(None)
                continue
            x = np.array([[float(z)], [0.0]], dtype=float)
            P = np.eye(2, dtype=float) * KALMAN_INIT_COV
            outputs.append(float(z))
            continue

        x = F @ x
        P = F @ P @ F.T + Q

        if z is not None:
            y = np.array([[float(z)]], dtype=float) - (H @ x)
            s = float((H @ P @ H.T + R)[0, 0])
            if s < 1e-12:
                s = 1e-12
            K = (P @ H.T) / s
            x = x + K @ y
            P = (I - K @ H) @ P

        outputs.append(float(x[0, 0]))

    return outputs


def select_rows_for_case(case_id, policy_mode, data_2d, data_tri, data_mon, strict_source_mode=None):
    if strict_source_mode in ("2d", "tri", "monster"):
        src_map = {
            "2d": data_2d,
            "tri": data_tri,
            "monster": data_mon,
        }
        rows = src_map[strict_source_mode].get(case_id, [])
        if rows:
            return copy_entries(rows), strict_source_mode
        return [], strict_source_mode

    # Normal mode follows the policy result first, but still keeps a fallback order
    # so downstream code does not crash if one source file is incomplete.
    if policy_mode == "tri":
        order = [("tri", data_tri), ("monster", data_mon), ("2d", data_2d)]
    elif policy_mode == "monster":
        order = [("monster", data_mon), ("tri", data_tri), ("2d", data_2d)]
    else:
        order = [("2d", data_2d), ("tri", data_tri), ("monster", data_mon)]

    for src_mode, src_data in order:
        rows = src_data.get(case_id, [])
        if rows:
            return copy_entries(rows), src_mode

    return [], ""


def apply_kalman_for_case(entries, coord_cols):
    for e in entries:
        for c in coord_cols:
            e["row"][f"{c}_kf"] = ""

    side_to_indices = {}
    for i, e in enumerate(entries):
        side_to_indices.setdefault(e["side"], [])
        side_to_indices[e["side"]].append(i)

    # Left and right views are filtered separately because they are
    # two independent time sequences in this dataset.
    for side in side_to_indices:
        idxs = side_to_indices[side]
        idxs.sort(key=lambda i: entries[i]["frame_idx"])

        for col in coord_cols:
            values = [safe_float(entries[i]["row"].get(col), default=None) for i in idxs]
            filtered = kalman_1d(values)

            for i, v in zip(idxs, filtered):
                key = f"{col}_kf"
                if v is None or not np.isfinite(v):
                    entries[i]["row"][key] = ""
                else:
                    entries[i]["row"][key] = round(float(v), ROUND_DIGITS)


def build_kalman_rows(
    policy_by_case,
    data_2d,
    data_tri,
    data_mon,
    verbose=True,
    only_policy_cases=False,
    strict_source_mode=None,
    skip_bad_cases=False,
):
    if only_policy_cases:
        all_cases = set(policy_by_case.keys())
    else:
        all_cases = set(policy_by_case.keys()) | set(data_2d.keys()) | set(data_tri.keys()) | set(data_mon.keys())
    out_rows = []

    used_by_source = {"2d": 0, "tri": 0, "monster": 0}
    bad_cases = 0
    missing_source_cases = 0

    for case_id in sorted(all_cases):
        policy_mode = policy_by_case.get(case_id, "None")

        # Bad case: policy says this case should not continue to Kalman.
        if is_bad_mode(policy_mode):
            bad_cases += 1
            if skip_bad_cases:
                continue
            out_rows.append({
                "case_id": case_id,
                "mode": policy_mode,
                "policy_mode": policy_mode,
                "source_mode": "",
                "side": "",
                "frame_idx": "",
            })
            continue

        entries, source_mode = select_rows_for_case(
            case_id,
            policy_mode,
            data_2d,
            data_tri,
            data_mon,
            strict_source_mode=strict_source_mode,
        )

        if not entries:
            missing_source_cases += 1
            if skip_bad_cases:
                continue
            out_rows.append({
                "case_id": case_id,
                "mode": policy_mode,
                "policy_mode": policy_mode,
                "source_mode": "",
                "side": "",
                "frame_idx": "",
            })
            continue

        used_by_source[source_mode] = used_by_source.get(source_mode, 0) + 1

        # 2D channels are always kept when present.
        # 3D channels are only added for TRI or MONSTER source rows.
        cols_2d = [c for c in KPT_2D_COLS if c in entries[0]["row"]]
        cols_3d = [c for c in KPT_3D_COLS if c in entries[0]["row"]] if source_mode in ("tri", "monster") else []
        coord_cols = cols_2d + cols_3d

        apply_kalman_for_case(entries, coord_cols)

        for e in entries:
            row_out = dict(e["row"])
            row_out["case_id"] = case_id
            row_out["mode"] = policy_mode
            row_out["policy_mode"] = policy_mode
            row_out["source_mode"] = source_mode
            row_out["side"] = e["side"]
            row_out["frame_idx"] = e["frame_idx"]
            out_rows.append(row_out)

    if verbose:
        print(f"[DONE] Kalman prepared rows: {len(out_rows)}")
        print(f"[INFO] Source used by case: tri={used_by_source.get('tri', 0)}, monster={used_by_source.get('monster', 0)}, 2d={used_by_source.get('2d', 0)}")
        if bad_cases > 0:
            print(f"[INFO] Bad cases skipped by policy (None/other): {bad_cases}")
        if missing_source_cases > 0:
            print(f"[WARNING] Cases with missing source rows: {missing_source_cases}")

    return out_rows


# ===============================================
# 3. save csv
# ===============================================
def merge_headers(*header_lists):
    merged = []
    seen = set()
    for headers in header_lists:
        for h in headers:
            if not h:
                continue
            key = "filename" if "filename" in str(h).lower() else h
            if key in seen:
                continue
            seen.add(key)
            merged.append(key)
    return merged


def save_csv(rows, base_headers, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta_cols = ["case_id", "mode", "policy_mode", "source_mode", "side", "frame_idx"]
    kf_cols = [f"{c}_kf" for c in (KPT_2D_COLS + KPT_3D_COLS)]

    raw_cols = []
    seen_raw = set()
    for c in base_headers:
        if c in meta_cols or c.endswith("_kf"):
            continue
        if c not in seen_raw:
            seen_raw.add(c)
            raw_cols.append(c)
    for r in rows:
        for c in r.keys():
            if c in meta_cols or c.endswith("_kf"):
                continue
            if c not in seen_raw:
                seen_raw.add(c)
                raw_cols.append(c)

    fieldnames = meta_cols + raw_cols + kf_cols

    def to_int(v, default=0):
        try:
            if v is None or v == "":
                return default
            return int(v)
        except (TypeError, ValueError):
            return default

    rows = list(rows)
    rows.sort(
        key=lambda r: (
            to_int(r.get("case_id", 0), 0),
            to_int(r.get("frame_idx", -1), -1),
            side_rank(r.get("side", "")),
        )
    )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Saved: {out_path}")


# ===============================================
# main
# ===============================================
def main():
    # load policy and source csv
    policy_by_case, _ = load_policy(POLICY_CSV_PATH)
    if not policy_by_case:
        print("[ERROR] Empty policy, exiting")
        return

    data_2d, headers_2d = load_detection_by_case(CSV_2D_PATH, tag="2D")
    data_tri, headers_tri = load_detection_by_case(CSV_TRI_PATH, tag="TRI-3D")
    data_mon, headers_mon = load_detection_by_case(CSV_MONSTER_PATH, tag="MONSTER-3D")

    # run kalman by policy mode
    out_rows = build_kalman_rows(policy_by_case, data_2d, data_tri, data_mon)
    if not out_rows:
        print("[ERROR] No output rows after filtering, exiting")
        return

    # save csv
    base_headers = merge_headers(headers_2d, headers_tri, headers_mon)
    save_csv(out_rows, base_headers, OUT_CSV_PATH)


if __name__ == "__main__":
    main()
