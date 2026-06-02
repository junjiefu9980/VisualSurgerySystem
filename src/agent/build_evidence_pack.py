"""
Build evidence pack for selected cases.
Only save files under output/agent.
"""
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# make local import ok
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent import event_detector
from src.agent import trajectory_visualizer


# ===============================================
# path
# ================================================
OUT_ROOT = REPO_ROOT / "output"
DET_DIR = OUT_ROOT / "detections"
EVAL_DIR = OUT_ROOT / "evaluation"

OUT_AGENT_DIR = OUT_ROOT / "agent"
FIG_DIR_NAME = "figures"

FINAL_CASE_TABLE_CSV = EVAL_DIR / "final_case_table.csv"
POLICY_CSV = EVAL_DIR / "policy.csv"

METHOD_CSV = {
    "2d": DET_DIR / "2d_results.csv",
    "tri": DET_DIR / "3d_tri_results.csv",
    "monster": DET_DIR / "3d_monster_results.csv",
}

METHOD_CSV_KF = {
    "2d": DET_DIR / "2d_kalman_results.csv",
    "tri": DET_DIR / "3d_tri_kalman_results.csv",
    "monster": DET_DIR / "3d_monster_kalman_results.csv",
}

METHOD_CSV_FALLBACK = {
    "2d": [DET_DIR / "2d_results.csv"],
    "tri": [DET_DIR / "3d_tri_results.csv"],
    "monster": [DET_DIR / "3d_monster_results.csv", DET_DIR / "[old]3d_monster_results.csv"],
}

# run config
AUTO_SELECT = True
CASE_ID = ""
N_PER_BUCKET = 1
TRI_PASS_PICK = 1
FPS = 30
K_MAD = 6.0
K_BREAK = 3.0
K_GATE = 6.0
TRACK_WINDOW = 30
MIN_LEN = 5
KPT_ID = "auto"
VIEW_SIDE = "both"
OUT_DIR = OUT_AGENT_DIR
SAVE_EVENTS_CSV = False


# ===============================================
# io tools
# ================================================
def load_rows(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def save_csv(rows, out_csv_path, cols):
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def normalize_case_id(case_id):
    s = str(case_id).strip()
    if s.isdigit():
        return str(int(s))
    return s


def case_sort_key(case_id):
    s = str(case_id)
    return int(s) if s.isdigit() else s


def parse_flag(x):
    s = str(x).strip().lower()
    if s in ("1", "true", "yes"):
        return 1
    if s in ("0", "false", "no"):
        return 0
    return None


def to_float(x, default=np.nan):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def normalize_mode(x):
    s = str(x).strip().lower()
    if s in ("tri", "monster", "2d", "none"):
        return "None" if s == "none" else s
    return "None"


def to_rel(path_obj):
    p = Path(path_obj)
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


# ===============================================
# case selection
# ================================================
def load_policy_mode_map():
    out = {}
    for r in load_rows(POLICY_CSV):
        cid = normalize_case_id(r.get("case_id", ""))
        if cid == "":
            continue
        out[cid] = normalize_mode(r.get("mode", "None"))
    return out


def build_case_table_map():
    rows = load_rows(FINAL_CASE_TABLE_CSV)
    out = {}
    for r in rows:
        cid = normalize_case_id(r.get("case_id", ""))
        if cid == "":
            continue
        out[cid] = r
    return out


def get_final_mode(case_row, policy_mode_map):
    if case_row:
        p1 = normalize_mode(case_row.get("policy_mode", ""))
        p2 = normalize_mode(case_row.get("chosen_code", ""))
        if p1 != "None":
            return p1
        if p2 != "None":
            return p2

    cid = normalize_case_id((case_row or {}).get("case_id", ""))
    if cid in policy_mode_map:
        return policy_mode_map[cid]
    return "None"


def select_cases_auto(case_table_map, policy_mode_map, n_per_bucket):
    rows = [case_table_map[cid] for cid in sorted(case_table_map.keys(), key=case_sort_key)]

    # Four case groups for report figure, one case each by default
    c1_2d_fail_none = []
    c2_fallback_2d = []
    c3_tri_pass = []
    c4_monster_rescue = []

    for r in rows:
        cid = normalize_case_id(r.get("case_id", ""))
        tri_ok = parse_flag(r.get("is_tri_ok", ""))
        mon_ok = parse_flag(r.get("is_monster_ok", ""))
        ok2d = parse_flag(r.get("is_2d_ok", ""))
        mode = get_final_mode(r, policy_mode_map)

        if ok2d == 0 or mode == "None":
            c1_2d_fail_none.append(cid)
            continue
        if mode == "2d" and tri_ok == 0 and mon_ok == 0:
            c2_fallback_2d.append(cid)
            continue
        if mode == "tri" and tri_ok == 1:
            c3_tri_pass.append(cid)
            continue
        if mode == "monster" and tri_ok == 0 and mon_ok == 1:
            c4_monster_rescue.append(cid)
            continue

    row_map = {normalize_case_id(r.get("case_id", "")): r for r in rows}

    def tri_fail_margin(r):
        reason = str(r.get("reason_tri", ""))
        if "valid_all<0.80" in reason:
            v = to_float(r.get("tri_valid_all"))
            return max(0.0, 0.80 - v) if np.isfinite(v) else 0.0
        if "disp_p95>260.0" in reason:
            v = to_float(r.get("tri_disp_p95"))
            return max(0.0, v - 260.0) if np.isfinite(v) else 0.0
        if "z_neg_ratio>0.10" in reason:
            v = to_float(r.get("tri_z_neg_ratio"))
            return max(0.0, v - 0.10) if np.isfinite(v) else 0.0
        if "z_p95" in reason:
            vals = [
                to_float(r.get("tri_z_p95_L1")),
                to_float(r.get("tri_z_p95_L2")),
                to_float(r.get("tri_z_p95_R1")),
                to_float(r.get("tri_z_p95_R2")),
            ]
            vals = [v for v in vals if np.isfinite(v)]
            if len(vals) == 0:
                return 0.0
            return max(vals) - 300.0
        return 0.0

    def monster_fail_margin(r):
        reason = str(r.get("reason_monster", ""))
        if "disp_p95>90.0" in reason:
            v = to_float(r.get("monster_disp_p95"))
            return max(0.0, v - 90.0) if np.isfinite(v) else 0.0
        if "valid_all<0.85" in reason:
            v = to_float(r.get("monster_valid_all"))
            return max(0.0, 0.85 - v) if np.isfinite(v) else 0.0
        if "z_p95" in reason:
            vals = [
                to_float(r.get("monster_z_p95_L1")),
                to_float(r.get("monster_z_p95_L2")),
                to_float(r.get("monster_z_p95_R1")),
                to_float(r.get("monster_z_p95_R2")),
            ]
            vals = [v for v in vals if np.isfinite(v)]
            if len(vals) == 0:
                return 0.0
            return max(vals) - 300.0
        return 0.0

    def score_c4(cid):
        r = row_map.get(cid, {})
        tri_disp = to_float(r.get("tri_disp_p95"))
        mon_disp = to_float(r.get("monster_disp_p95"))
        tri_valid = to_float(r.get("tri_valid_all"))
        mon_valid = to_float(r.get("monster_valid_all"))
        gap_disp = tri_disp - mon_disp if np.isfinite(tri_disp) and np.isfinite(mon_disp) else -1e9
        gap_valid = mon_valid - tri_valid if np.isfinite(mon_valid) and np.isfinite(tri_valid) else -1e9
        fail = tri_fail_margin(r)
        return (fail, gap_disp, gap_valid)

    def score_c2(cid):
        r = row_map.get(cid, {})
        tri_disp = to_float(r.get("tri_disp_p95"))
        mon_disp = to_float(r.get("monster_disp_p95"))
        fail_tri = tri_fail_margin(r)
        fail_mon = monster_fail_margin(r)
        return (fail_mon, fail_tri, max(tri_disp, mon_disp))

    def score_c3(cid):
        r = row_map.get(cid, {})
        tri_disp = to_float(r.get("tri_disp_p95"))
        tri_valid = to_float(r.get("tri_valid_all"))
        if not np.isfinite(tri_valid):
            tri_valid = 0.0
        if not np.isfinite(tri_disp):
            tri_disp = 1e6
        score = tri_valid - 0.001 * tri_disp
        return (score, tri_valid, -tri_disp)

    def score_c1(cid):
        r = row_map.get(cid, {})
        f1 = to_float(r.get("2d_f1"))
        m50 = to_float(r.get("2d_mAP50"))
        return (-f1, -m50)

    c1_2d_fail_none = sorted(c1_2d_fail_none, key=lambda x: score_c1(x), reverse=True)
    c2_fallback_2d = sorted(c2_fallback_2d, key=lambda x: score_c2(x), reverse=True)
    c3_tri_pass = sorted(c3_tri_pass, key=lambda x: score_c3(x), reverse=True)
    c4_monster_rescue = sorted(c4_monster_rescue, key=lambda x: score_c4(x), reverse=True)

    pick = []
    n = int(max(1, n_per_bucket))
    pick += c1_2d_fail_none[:n]
    pick += c2_fallback_2d[:n]
    pick += c3_tri_pass[: max(1, min(TRI_PASS_PICK, len(c3_tri_pass)))]
    pick += c4_monster_rescue[:n]

    seen = set()
    selected = []
    for cid in pick:
        if cid in seen:
            continue
        seen.add(cid)
        selected.append(cid)

    if len(selected) == 0 and len(rows) > 0:
        selected = [normalize_case_id(rows[0].get("case_id", ""))]

    bucket_map = {
        "c1_2d_fail_none": c1_2d_fail_none,
        "c2_fallback_2d": c2_fallback_2d,
        "c3_tri_pass": c3_tri_pass,
        "c4_monster_rescue": c4_monster_rescue,
    }

    reason_map = {}
    for cid in selected:
        r = row_map.get(cid, {})
        mode = get_final_mode(r, policy_mode_map)
        bucket = get_case_bucket(cid, bucket_map)
        if bucket == "c1_2d_fail_none":
            reason_map[cid] = (
                f"C1 2d fail: reason_2d={r.get('reason_2d', '')}, 2d_f1={r.get('2d_f1')}, policy_mode={mode}"
            )
        elif bucket == "c2_fallback_2d":
            reason_map[cid] = (
                f"C2 fallback 2d: tri fail ({r.get('reason_tri', '')}), "
                f"monster fail ({r.get('reason_monster', '')}), 2d_f1={r.get('2d_f1')}"
            )
        elif bucket == "c3_tri_pass":
            reason_map[cid] = (
                f"C3 tri pass: tri_valid_all={r.get('tri_valid_all')}, tri_disp_p95={r.get('tri_disp_p95')}, 2d_f1={r.get('2d_f1')}"
            )
        else:
            reason_map[cid] = (
                f"C4 monster rescue: tri fail ({r.get('reason_tri', '')}); "
                f"tri_disp_p95={r.get('tri_disp_p95')} -> monster_disp_p95={r.get('monster_disp_p95')}"
            )

    return selected, bucket_map, reason_map


def get_case_bucket(case_id, bucket_map):
    cid = normalize_case_id(case_id)
    if cid in bucket_map.get("c1_2d_fail_none", []):
        return "c1_2d_fail_none"
    if cid in bucket_map.get("c2_fallback_2d", []):
        return "c2_fallback_2d"
    if cid in bucket_map.get("c3_tri_pass", []):
        return "c3_tri_pass"
    if cid in bucket_map.get("c4_monster_rescue", []):
        return "c4_monster_rescue"
    return "other"


# ===============================================
# event summary
# ================================================
def p95(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.percentile(arr, 95))


def sum_event_duration_s(events, fps):
    total = 0.0
    for e in events:
        s = int(e["start_frame"])
        t = int(e["end_frame"])
        total += max(0.0, (t - s + 1) / float(fps))
    return total


def prepare_events_for_json(events):
    out = []
    for e in events:
        ee = dict(e)
        ev = ee.get("evidence", {})
        if isinstance(ev, str):
            try:
                ev = json.loads(ev)
            except Exception:
                ev = {"text": ev}
        ee["evidence"] = ev
        out.append(ee)
    return out


def build_evidence_row(case_id, side, method, event):
    ev_type = str(event.get("event_type", ""))
    s_t = float(event.get("start_time_s", 0.0))
    e_t = float(event.get("end_time_s", 0.0))
    ev = event.get("evidence", {})
    if isinstance(ev, str):
        try:
            ev = json.loads(ev)
        except Exception:
            ev = {}

    th = ev.get("threshold", {}) if isinstance(ev, dict) else {}
    st = ev.get("stats", {}) if isinstance(ev, dict) else {}

    metric = ""
    relation = ""
    value = None
    threshold = None

    if ev_type == "JUMP_OUTLIER":
        metric = "disp"
        relation = ">"
        value = st.get("max_disp", None)
        threshold = th.get("value", None)
    elif ev_type == "JITTER_SPIKE":
        metric = "jerk"
        relation = ">"
        value = st.get("max_jerk", None)
        threshold = th.get("value", None)
    elif ev_type == "STALL_STOP":
        metric = "disp"
        relation = "<"
        value = st.get("min_disp", None)
        threshold = th.get("value", None)
    elif ev_type == "MISSING_LOW_VALID":
        metric = "missing_ratio"
        relation = ">"
        value = st.get("missing_ratio", None)
        threshold = 0.0

    def fmt(v):
        if v is None:
            return "None"
        try:
            fv = float(v)
            if not np.isfinite(fv):
                return "None"
            return f"{fv:.4f}"
        except Exception:
            return "None"

    chain_text = f"{s_t:.2f}-{e_t:.2f}s {metric} {relation} {fmt(threshold)}, value={fmt(value)}"

    return {
        "case_id": normalize_case_id(case_id),
        "side": str(side),
        "method": str(method),
        "start_frame": int(event.get("start_frame", 0)),
        "end_frame": int(event.get("end_frame", 0)),
        "start_time_s": round(s_t, 4),
        "end_time_s": round(e_t, 4),
        "event_type": ev_type,
        "severity": float(event.get("severity", 0.0)),
        "metric": metric,
        "relation": relation,
        "value": fmt(value),
        "threshold": fmt(threshold),
        "evidence_chain": chain_text,
    }


def build_evidence_rows(case_id, side, method, events):
    rows = []
    for e in events:
        rows.append(build_evidence_row(case_id, side, method, e))
    return rows


def write_events_csv(case_id, all_events, out_dir):
    out_path = Path(out_dir) / f"events_{normalize_case_id(case_id)}.csv"
    event_detector.save_events_csv(all_events, out_path)
    return out_path


def resolve_method_csv(method, prefer_kf=True):
    cands = []
    if prefer_kf:
        cands.append(METHOD_CSV_KF.get(method))
    cands.extend(METHOD_CSV_FALLBACK.get(method, [METHOD_CSV.get(method)]))
    for p in cands:
        if p is None:
            continue
        pp = Path(p)
        if pp.exists():
            return pp
    return Path(METHOD_CSV.get(method))


def get_case_scenario(case_row, final_mode):
    ok2d = parse_flag((case_row or {}).get("is_2d_ok", ""))
    tri_ok = parse_flag((case_row or {}).get("is_tri_ok", ""))
    mon_ok = parse_flag((case_row or {}).get("is_monster_ok", ""))
    mode = normalize_mode(final_mode)

    if ok2d == 0 or mode == "None":
        return "c1_2d_fail_none"
    if mode == "2d" and tri_ok == 0 and mon_ok == 0:
        return "c2_fallback_2d"
    if mode == "tri" and tri_ok == 1:
        return "c3_tri_pass"
    if mode == "monster" and tri_ok == 0 and mon_ok == 1:
        return "c4_monster_rescue"
    return "other"


def get_methods(final_mode):
    mode = normalize_mode(final_mode)
    if mode in ("tri", "monster", "2d"):
        return [mode]
    return ["2d"]


def method_ok(case_row, method):
    if method == "2d":
        return parse_flag((case_row or {}).get("is_2d_ok", "")) == 1
    if method == "tri":
        return parse_flag((case_row or {}).get("is_tri_ok", "")) == 1
    if method == "monster":
        return parse_flag((case_row or {}).get("is_monster_ok", "")) == 1
    return False


def method_reason(case_row, method):
    if method == "2d":
        return str((case_row or {}).get("reason_2d", ""))
    if method == "tri":
        return str((case_row or {}).get("reason_tri", ""))
    if method == "monster":
        return str((case_row or {}).get("reason_monster", ""))
    return ""


def make_event(case_id, method, frames, s_idx, e_idx, fps, event_type, thr_value, metric_name):
    s_frame = int(frames[s_idx])
    e_frame = int(frames[e_idx])
    return {
        "case_id": normalize_case_id(case_id),
        "method": method,
        "start_frame": s_frame,
        "end_frame": e_frame,
        "start_time_s": round(s_frame / float(fps), 4),
        "end_time_s": round(e_frame / float(fps), 4),
        "event_type": event_type,
        "severity": 1.0,
        "evidence": {
            "threshold": {"value": None if thr_value is None else float(thr_value)},
            "stats": {"metric": metric_name},
        },
    }


def build_reason_events(case_id, method, det, reason_text, fps, min_len):
    reason = str(reason_text or "").strip()
    if reason in ("", "pass", "None"):
        return []

    track = det.get("track", {})
    metrics = det.get("metrics", {})
    frames = np.asarray(track.get("frames", []), dtype=int)
    if len(frames) == 0:
        return []

    seg_min = max(3, int(min_len))
    events = []

    def add_from_mask(mask, event_type, thr_value, metric_name):
        if len(mask) != len(frames):
            return
        segs = event_detector.mask_to_segments(frames, np.asarray(mask, dtype=bool), min_len=seg_min)
        for s, e in segs:
            events.append(make_event(case_id, method, frames, s, e, fps, event_type, thr_value, metric_name))

    # Use gate reason to find the bad window if raw event is empty
    m = re.search(r">([0-9.]+)", reason)
    reason_thr = float(m.group(1)) if m else None

    if "valid_all<" in reason:
        valid = np.asarray(track.get("valid", []), dtype=bool)
        add_from_mask(~valid, "MISSING_LOW_VALID", None, "valid")

    if len(events) == 0 and "disp_p95>" in reason:
        disp = np.asarray(metrics.get("disp", []), dtype=float)
        if reason_thr is None:
            fin = disp[np.isfinite(disp)]
            reason_thr = float(np.percentile(fin, 90)) if fin.size > 0 else None
        mask = np.isfinite(disp) & (disp > float(reason_thr)) if reason_thr is not None else np.zeros(len(frames), dtype=bool)
        add_from_mask(mask, "JUMP_OUTLIER", reason_thr, "disp")

    if len(events) == 0 and "z_neg_ratio>" in reason and int(track.get("dim", 2)) >= 3:
        z = np.asarray(track.get("coords", np.zeros((0, 3))))[:, 2] if len(track.get("coords", [])) > 0 else np.asarray([])
        if len(z) == len(frames):
            add_from_mask(np.isfinite(z) & (z < 0.0), "Z_NEGATIVE", 0.0, "z")

    if len(events) == 0 and "z_p95" in reason and int(track.get("dim", 2)) >= 3:
        z = np.asarray(track.get("coords", np.zeros((0, 3))))[:, 2] if len(track.get("coords", [])) > 0 else np.asarray([])
        if reason_thr is None:
            reason_thr = 300.0
        if len(z) == len(frames):
            add_from_mask(np.isfinite(z) & (z > float(reason_thr)), "Z_HIGH", reason_thr, "z")

    # Fallback: take motion outlier
    if len(events) == 0:
        disp = np.asarray(metrics.get("disp", []), dtype=float)
        fin = disp[np.isfinite(disp)]
        if fin.size > 0:
            thr = float(np.percentile(fin, 90))
            add_from_mask(np.isfinite(disp) & (disp > thr), "JUMP_OUTLIER", thr, "disp")

    return events


# ===============================================
# run one case
# ================================================
def detect_one_method(
    case_id,
    method,
    side,
    fps,
    k_mad,
    min_len,
    kpt_id,
    continuity,
    k_gate,
    track_window,
    prefer_kf=True,
):
    csv_path = resolve_method_csv(method, prefer_kf=prefer_kf)
    csv_path = Path(csv_path)

    if not csv_path.exists():
        return {
            "case_id": normalize_case_id(case_id),
            "method": method,
            "csv_path": str(csv_path),
            "kpt_id": str(kpt_id),
            "view_side": str(side),
            "track": {
                "frames": np.asarray([], dtype=int),
                "coords": np.asarray([], dtype=float),
                "valid": np.asarray([], dtype=bool),
                "conf": np.asarray([], dtype=float),
                "dim": 3 if method in ("tri", "monster") else 2,
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
            "fps": float(fps),
        }

    det = event_detector.detect_case_events(
        case_id=case_id,
        method=method,
        fps=fps,
        k_mad=k_mad,
        min_len=min_len,
        kpt_id=kpt_id,
        csv_path=csv_path,
        view_side=side,
        continuity=continuity,
        k_gate=k_gate,
        track_window=track_window,
    )
    det["csv_path"] = str(csv_path)

    # If kf file exists but this case is empty in it, fallback to raw file.
    track_frames = det.get("track", {}).get("frames", [])
    if prefer_kf and len(track_frames) == 0:
        raw_csv = Path(resolve_method_csv(method, prefer_kf=False))
        if raw_csv.exists() and raw_csv.resolve() != csv_path.resolve():
            det = event_detector.detect_case_events(
                case_id=case_id,
                method=method,
                fps=fps,
                k_mad=k_mad,
                min_len=min_len,
                kpt_id=kpt_id,
                csv_path=raw_csv,
                view_side=side,
                continuity=continuity,
                k_gate=k_gate,
                track_window=track_window,
            )
            det["csv_path"] = str(raw_csv)

    det["fps"] = float(fps)
    det["view_side"] = str(side)
    return det


def run_one_side(
    case_id,
    side,
    case_row,
    final_mode,
    fps,
    k_mad,
    min_len,
    out_dir,
    kpt_id,
    k_break,
    continuity,
    k_gate,
    track_window,
):
    scenario = get_case_scenario(case_row, final_mode)
    methods = get_methods(final_mode)

    auto_side_kpt = str(kpt_id).strip()
    if auto_side_kpt in ("", "auto"):
        auto_side_kpt = f"auto_{side}"

    det_2d_anchor = detect_one_method(
        case_id=case_id,
        method="2d",
        side=side,
        fps=fps,
        k_mad=k_mad,
        min_len=min_len,
        kpt_id=auto_side_kpt,
        continuity=continuity,
        k_gate=k_gate,
        track_window=track_window,
        prefer_kf=(scenario != "c1_2d_fail_none"),
    )
    anchor_kpt = str(det_2d_anchor.get("kpt_id", "")).strip()
    if anchor_kpt == "":
        anchor_kpt = auto_side_kpt

    side_entry = {
        "case_id": normalize_case_id(case_id),
        "side": str(side),
        "final_mode": final_mode,
        "scenario": scenario,
        "chosen_method": methods[0] if len(methods) > 0 else normalize_mode(final_mode),
        "anchor_kpt": anchor_kpt,
        "gate_reasons": {
            "reason_2d": (case_row or {}).get("reason_2d", ""),
            "reason_tri": (case_row or {}).get("reason_tri", ""),
            "reason_monster": (case_row or {}).get("reason_monster", ""),
        },
        "methods": {},
    }

    all_events = []
    evidence_rows = []

    method_data = {}

    for method in methods:
        # 2d gives the anchor kpt for this side; 3d follows it
        if method == "2d":
            det = det_2d_anchor
        else:
            det = detect_one_method(
                case_id=case_id,
                method=method,
                side=side,
                fps=fps,
                k_mad=k_mad,
                min_len=min_len,
                kpt_id=anchor_kpt,
                continuity=continuity,
                k_gate=k_gate,
                track_window=track_window,
                prefer_kf=(scenario != "c1_2d_fail_none"),
            )

        gate_ok = method_ok(case_row, method)
        reason = method_reason(case_row, method)

        if gate_ok:
            # Pass route should be clean in figure
            method_events = []
        else:
            method_events = list(det.get("events", []))
            if len(method_events) == 0:
                method_events = build_reason_events(
                    case_id=case_id,
                    method=method,
                    det=det,
                    reason_text=reason,
                    fps=fps,
                    min_len=min_len,
                )

        det["events"] = method_events
        det["gate_pass"] = bool(gate_ok)
        method_data[method] = {
            "det": det,
            "gate_ok": bool(gate_ok),
            "reason": reason,
            "events": method_events,
        }

        all_events.extend(method_events)
        evidence_rows.extend(build_evidence_rows(case_id, side, method, method_events))

    axis_2d = {
        "xlim": trajectory_visualizer.FIXED_2D_XLIM,
        "ylim": trajectory_visualizer.FIXED_2D_YLIM,
    }
    plot_track_map = {}
    for method in methods:
        det = method_data[method]["det"]
        plot_track_map[method] = det.get("track", {})

    # Build same 3d view scale for this case side
    coords_3d = []
    for method in methods:
        track = plot_track_map.get(method, {})
        if int(track.get("dim", 2)) < 3:
            continue
        coords = np.asarray(track.get("coords", []), dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 3 or len(coords) == 0:
            continue
        coords_3d.append(coords[:, :3])
    axis_3d = trajectory_visualizer.build_cube_limits_from_coords(coords_3d, pad_ratio=0.08, q_low=2.0, q_high=98.0)
    for method in methods:
        pack = method_data[method]
        det = pack["det"]
        method_events = pack["events"]
        gate_ok = pack["gate_ok"]
        reason = pack["reason"]
        det_plot = dict(det)
        det_plot["track"] = plot_track_map.get(method, det.get("track", {}))

        figs = trajectory_visualizer.save_method_figures(
            det_plot,
            out_dir=out_dir,
            save_timeline=True,
            k_break=k_break,
            axis_2d=axis_2d,
            axis_3d=axis_3d,
        )

        metrics = det.get("metrics", {})
        disp_p95 = p95(metrics.get("disp", []))
        jerk_p95 = p95(metrics.get("jerk", []))
        valid_ratio = float(metrics.get("valid_ratio", 0.0))

        side_entry["methods"][method] = {
            "events": prepare_events_for_json(method_events),
            "figures": [to_rel(p) for p in figs],
            "key_metrics": {
                "track_frames": int(len(det.get("track", {}).get("frames", []))),
                "valid_ratio": round(valid_ratio, 6),
                "disp_p95": None if disp_p95 is None else round(disp_p95, 6),
                "jerk_p95": None if jerk_p95 is None else round(jerk_p95, 6),
                "event_count": int(len(method_events)),
                "red_total_s": round(sum_event_duration_s(method_events, fps), 6),
            },
            "kpt_id": det.get("kpt_id", ""),
            "kpt_match_anchor": int(str(det.get("kpt_id", "")) == str(anchor_kpt)),
            "gate_pass": int(bool(det_plot.get("gate_pass", False))),
            "gate_reason": reason,
        }

    all_events.sort(key=lambda r: (r["method"], int(r["start_frame"]), int(r["end_frame"])))
    return side_entry, all_events, evidence_rows


def run_one_case(
    case_id,
    case_row,
    final_mode,
    fps,
    k_mad,
    min_len,
    out_dir,
    kpt_id,
    run_sides,
    k_break=3.0,
    continuity=True,
    k_gate=6.0,
    track_window=30,
    save_events_csv=False,
):
    case_entries = []
    all_events = []
    all_evidence_rows = []

    for side in run_sides:
        side_entry, side_events, side_evidence_rows = run_one_side(
            case_id=case_id,
            side=side,
            case_row=case_row,
            final_mode=final_mode,
            fps=fps,
            k_mad=k_mad,
            min_len=min_len,
            out_dir=out_dir,
            kpt_id=kpt_id,
            k_break=k_break,
            continuity=continuity,
            k_gate=k_gate,
            track_window=track_window,
        )
        case_entries.append(side_entry)
        all_events.extend(side_events)
        all_evidence_rows.extend(side_evidence_rows)

    all_events.sort(key=lambda r: (r["method"], int(r["start_frame"]), int(r["end_frame"])))
    events_csv_path = None
    if save_events_csv:
        events_csv_path = write_events_csv(case_id, all_events, out_dir)

    return case_entries, all_events, all_evidence_rows, events_csv_path


# ===============================================
# run
# ================================================
def run(
    case_id="",
    auto_select=True,
    n_per_bucket=1,
    fps=30,
    k_mad=6.0,
    k_break=3.0,
    k_gate=6.0,
    track_window=30,
    continuity=True,
    min_len=5,
    out_dir=OUT_AGENT_DIR,
    kpt_id="auto",
    view_side="both",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / FIG_DIR_NAME).mkdir(parents=True, exist_ok=True)

    print("\n", "-" * 5, "Step 1: Load case tables", "-" * 5)
    case_table_map = build_case_table_map()
    policy_mode_map = load_policy_mode_map()

    if len(case_table_map) == 0:
        print(f"[ERROR] missing or empty: {FINAL_CASE_TABLE_CSV}")
        return

    # choose cases
    selected_cases = []
    select_reason_map = {}
    bucket_map = {
        "c1_2d_fail_none": [],
        "c2_fallback_2d": [],
        "c3_tri_pass": [],
        "c4_monster_rescue": [],
    }

    if str(case_id).strip() != "":
        selected_cases = [normalize_case_id(case_id)]
        print(f"[INFO] selected case: {selected_cases}")
    elif auto_select:
        selected_cases, bucket_map, select_reason_map = select_cases_auto(case_table_map, policy_mode_map, n_per_bucket)
        print(f"[INFO] auto selected: {selected_cases}")
    else:
        print("[ERROR] no case selected")
        return

    if str(view_side).strip().lower() in ("left", "right"):
        run_sides = [str(view_side).strip().lower()]
    else:
        run_sides = ["left", "right"]

    print("\n", "-" * 5, "Step 2: Run case by case", "-" * 5)
    all_case_entries = []
    summary_rows = []
    evidence_chain_rows = []

    for cid in selected_cases:
        row = case_table_map.get(normalize_case_id(cid), {})
        final_mode = get_final_mode(row, policy_mode_map)

        case_entries, all_events, case_evidence_rows, events_csv = run_one_case(
            case_id=cid,
            case_row=row,
            final_mode=final_mode,
            fps=fps,
            k_mad=k_mad,
            min_len=min_len,
            out_dir=out_dir,
            kpt_id=kpt_id,
            run_sides=run_sides,
            k_break=k_break,
            continuity=continuity,
            k_gate=k_gate,
            track_window=track_window,
            save_events_csv=SAVE_EVENTS_CSV,
        )
        all_case_entries.extend(case_entries)
        evidence_chain_rows.extend(case_evidence_rows)

        for ce in case_entries:
            ce["selection_reason"] = select_reason_map.get(normalize_case_id(cid), "")
            side = str(ce.get("side", ""))
            chosen_method = str(ce.get("chosen_method", "")).strip()
            method_data = ce.get("methods", {})
            if chosen_method == "" or chosen_method not in method_data:
                chosen_method = next(iter(method_data.keys()), "")
            chosen = method_data.get(chosen_method, {})
            chosen_events = chosen.get("events", [])
            chosen_metrics = chosen.get("key_metrics", {}) or {}
            chosen_figures = chosen.get("figures", [])

            max_sev = 0.0
            top_event = ""
            for e in chosen_events:
                sev = float(e.get("severity", 0.0))
                if sev >= max_sev:
                    max_sev = sev
                    top_event = str(e.get("event_type", ""))

            traj_png = ""
            timeline_png = ""
            for fig in chosen_figures:
                if str(fig).endswith("_traj.png"):
                    traj_png = str(fig)
                elif str(fig).endswith("_timeline.png"):
                    timeline_png = str(fig)

            summary_rows.append(
                {
                    "case_id": normalize_case_id(cid),
                    "side": side,
                    "bucket": get_case_bucket(cid, bucket_map),
                    "final_mode": final_mode,
                    "chosen_method": chosen_method,
                    "anchor_kpt": ce.get("anchor_kpt", ""),
                    "chosen_event_count": int(len(chosen_events)),
                    "chosen_red_total_s": round(sum_event_duration_s(chosen_events, fps), 6),
                    "chosen_valid_ratio": chosen_metrics.get("valid_ratio"),
                    "chosen_disp_p95": chosen_metrics.get("disp_p95"),
                    "max_severity": round(max_sev, 4),
                    "chosen_top_event_type": top_event,
                    "select_reason": select_reason_map.get(normalize_case_id(cid), ""),
                    "reason_2d": ce.get("gate_reasons", {}).get("reason_2d", ""),
                    "reason_tri": ce.get("gate_reasons", {}).get("reason_tri", ""),
                    "reason_monster": ce.get("gate_reasons", {}).get("reason_monster", ""),
                    "chosen_traj_png": traj_png,
                    "chosen_timeline_png": timeline_png,
                    "events_csv": "" if events_csv is None else to_rel(events_csv),
                }
            )

        if events_csv is None:
            print(f"[DONE] case={cid}, events={len(all_events)}")
        else:
            print(f"[DONE] case={cid}, events={len(all_events)}, file={to_rel(events_csv)}")

    print("\n", "-" * 5, "Step 3: Save evidence pack", "-" * 5)
    evidence_pack = {
        "meta": {
            "fps": float(fps),
            "k_mad": float(k_mad),
            "k_break": float(k_break),
            "k_gate": float(k_gate),
            "track_window": int(track_window),
            "continuity": bool(continuity),
            "run_sides": run_sides,
            "min_len": int(min_len),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "route_mode": "final_only",
            "note": "trajectory-quality only",
        },
        "cases": all_case_entries,
        "selected_case_reasons": select_reason_map,
    }

    evidence_json = out_dir / "evidence_pack.json"
    with open(evidence_json, "w", encoding="utf-8") as f:
        json.dump(evidence_pack, f, ensure_ascii=False, indent=2)

    summary_csv = out_dir / "summary.csv"
    summary_cols = [
        "case_id",
        "side",
        "bucket",
        "final_mode",
        "chosen_method",
        "anchor_kpt",
        "chosen_event_count",
        "chosen_red_total_s",
        "chosen_valid_ratio",
        "chosen_disp_p95",
        "max_severity",
        "chosen_top_event_type",
        "select_reason",
        "reason_2d",
        "reason_tri",
        "reason_monster",
        "chosen_traj_png",
        "chosen_timeline_png",
        "events_csv",
    ]
    save_csv(summary_rows, summary_csv, summary_cols)

    evidence_chain_csv = out_dir / "evidence_chain.csv"
    # This csv keeps the time window evidence used in feedback
    chain_cols = [
        "case_id",
        "side",
        "method",
        "start_frame",
        "end_frame",
        "start_time_s",
        "end_time_s",
        "event_type",
        "severity",
        "metric",
        "relation",
        "value",
        "threshold",
        "evidence_chain",
    ]
    save_csv(evidence_chain_rows, evidence_chain_csv, chain_cols)

    print(f"[DONE] saved: {evidence_json}")
    print(f"[DONE] saved: {summary_csv}")
    print(f"[DONE] saved: {evidence_chain_csv}")


# ===============================================
# main
# ================================================
def main():
    run(
        case_id=CASE_ID,
        auto_select=AUTO_SELECT,
        n_per_bucket=N_PER_BUCKET,
        fps=FPS,
        k_mad=K_MAD,
        k_break=K_BREAK,
        k_gate=K_GATE,
        track_window=TRACK_WINDOW,
        continuity=True,
        min_len=MIN_LEN,
        out_dir=OUT_DIR,
        kpt_id=KPT_ID,
        view_side=VIEW_SIDE,
    )


if __name__ == "__main__":
    main()
