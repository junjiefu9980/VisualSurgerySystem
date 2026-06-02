"""
Build final summary table, final case table and llm input.
"""
import csv
import json
import math
from datetime import datetime
from pathlib import Path
import numpy as np


# ===============================================
# policy import
# ================================================
try:
    from .policy import GatingPolicy
except ImportError:
    from policy import GatingPolicy  # type: ignore


# ===============================================
# config
# ================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "output" / "evaluation"
ROUND_DIGITS = 6


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


# fmt
# ================================================
def fmt_num(x):
    if x is None:
        return ""
    return round(float(x), ROUND_DIGITS)


# fmt number or None text
# ================================================
def fmt_num_or_none(x):
    v = fmt_num(x)
    return v if v != "" else "None"


# normalize mode
# ================================================
def normalize_mode(mode):
    m = str(mode or "").strip().lower()
    if m in ("tri", "monster", "2d"):
        return m
    return "None"


# load rows
# ================================================
def load_rows(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


# load eval by case + ALL
# ================================================
def load_eval_by_case_and_all(csv_path):
    by_case = {}
    all_row = {}
    for row in load_rows(csv_path):
        cid = str(row.get("case_id", "")).strip()
        if cid.upper() == "ALL":
            all_row = row
            continue
        if cid == "":
            continue
        by_case[cid] = row
    return by_case, all_row


# load policy
# ================================================
def load_policy(policy_csv_path):
    by_case = {}
    counts = {"tri": 0, "monster": 0, "2d": 0, "None": 0}
    bad_ids = []

    for row in load_rows(policy_csv_path):
        cid = str(row.get("case_id", "")).strip()
        if cid == "":
            continue
        mode = normalize_mode(row.get("mode", ""))
        by_case[cid] = mode
        counts[mode] += 1
        if mode == "None":
            bad_ids.append(cid)

    bad_ids = sorted(set(bad_ids), key=lambda x: int(x) if x.isdigit() else x)
    return by_case, counts, bad_ids


# load model gate
# ================================================
def load_model_gate(model_csv_path):
    rows = load_rows(model_csv_path)
    return rows[-1] if rows else {}


# load kalman per-case
# ================================================
def load_kalman_case(kalman_csv_path):
    rows = load_rows(kalman_csv_path)
    by_case_method = {}

    for r in rows:
        cid = str(r.get("case_id", "")).strip()
        method = str(r.get("method", "")).strip()
        if cid == "" or method == "":
            continue
        by_case_method[(cid, method)] = r

    return rows, by_case_method


# aggregate kalman by method
# ================================================
def aggregate_kalman_by_method(kalman_rows):
    metric_keys = [
        "disp_p95_raw",
        "disp_p95_kf",
        "improve_disp_p95",
        "jitter_med_raw",
        "jitter_med_kf",
        "improve_jitter_med",
    ]

    tmp = {}
    for r in kalman_rows:
        method = str(r.get("method", "")).strip()
        if method == "":
            continue
        tmp.setdefault(method, {k: [] for k in metric_keys})
        for k in metric_keys:
            v = safe_float(r.get(k), None)
            if v is not None:
                tmp[method][k].append(v)

    out = {}
    rows = []
    for method in sorted(tmp.keys()):
        row = {"method": method}
        for k in metric_keys:
            vals = tmp[method][k]
            row[k] = fmt_num(sum(vals) / len(vals)) if vals else ""
        out[method] = row
        rows.append(row)

    return out, rows


# 2d gate
# ================================================
def eval_2d_reason(policy_obj, row):
    f1 = policy_obj.safe_float((row or {}).get("f1"), 0.0)
    map50 = policy_obj.safe_float((row or {}).get("mAP50"), 0.0)
    if f1 < policy_obj.TH_2D["f1_min"]:
        return False, "f1<0.85"
    if map50 < policy_obj.TH_2D["map50_min"]:
        return False, "mAP50<0.80"
    return True, "pass"


# tri gate
# ================================================
def eval_tri_reason(policy_obj, row):
    valid_all = policy_obj.safe_float((row or {}).get("valid_all"), 0.0)
    disp_p95 = policy_obj.safe_float((row or {}).get("disp_p95"), 9999.0)
    z_neg = policy_obj.safe_float((row or {}).get("z_neg_ratio"), 1.0)

    if valid_all < policy_obj.TH_TRI["valid_all_min"]:
        return False, f"valid_all<{policy_obj.TH_TRI['valid_all_min']:.2f}"
    if disp_p95 > policy_obj.TH_TRI["disp_p95_max"]:
        return False, f"disp_p95>{policy_obj.TH_TRI['disp_p95_max']}"
    if z_neg > policy_obj.TH_TRI["z_neg_ratio_max"]:
        return False, f"z_neg_ratio>{policy_obj.TH_TRI['z_neg_ratio_max']:.2f}"

    for k in ["z_p95_L1", "z_p95_L2", "z_p95_R1", "z_p95_R2"]:
        if policy_obj.safe_float((row or {}).get(k), 9999.0) > policy_obj.TH_TRI["z_p95_max"]:
            return False, f"{k}>{policy_obj.TH_TRI['z_p95_max']}"

    return True, "pass"


# monster gate
# ================================================
def eval_monster_reason(policy_obj, row):
    valid_all = policy_obj.safe_float((row or {}).get("valid_all"), 0.0)
    disp_p95 = policy_obj.safe_float((row or {}).get("disp_p95"), 9999.0)

    if valid_all < policy_obj.TH_MONSTER["valid_all_min"]:
        return False, f"valid_all<{policy_obj.TH_MONSTER['valid_all_min']:.2f}"
    if disp_p95 > policy_obj.TH_MONSTER["disp_p95_max"]:
        return False, f"disp_p95>{policy_obj.TH_MONSTER['disp_p95_max']}"

    for k in ["z_p95_L1", "z_p95_L2", "z_p95_R1", "z_p95_R2"]:
        if policy_obj.safe_float((row or {}).get(k), 9999.0) > policy_obj.TH_MONSTER["z_p95_max"]:
            return False, f"{k}>{policy_obj.TH_MONSTER['z_p95_max']}"

    return True, "pass"


# build final case table
# ================================================
def build_final_case_table(policy_by_case, eval2d_by_case, tri_by_case, mon_by_case, kal_case_map):
    policy_obj = GatingPolicy()

    all_case_ids = set(policy_by_case.keys()) | set(eval2d_by_case.keys()) | set(tri_by_case.keys()) | set(mon_by_case.keys())
    rows = []

    for cid in sorted(all_case_ids, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        cid = str(cid)

        row2d = eval2d_by_case.get(cid, {})
        rowtri = tri_by_case.get(cid, {})
        rowmon = mon_by_case.get(cid, {})

        ok2d, reason2d = eval_2d_reason(policy_obj, row2d)
        oktri, reasontri = eval_tri_reason(policy_obj, rowtri)
        okmon, reasonmon = eval_monster_reason(policy_obj, rowmon)

        mode = normalize_mode(policy_by_case.get(cid, "None"))

        # force None when 2d fail
        if not ok2d:
            mode = "None"

        chosen_code = mode if mode in ("2d", "tri", "monster") else "None"

        # keep gate value for tri and monster
        show_tri_gate = mode in ("2d", "tri", "monster")
        show_mon_gate = mode in ("2d", "monster")
        show_tri_metric = mode in ("2d", "tri", "monster")
        show_mon_metric = mode in ("2d", "monster")

        # hidden style
        hidden_tri = "None" if mode == "None" else ""
        hidden_mon = "None" if mode in ("None", "tri") else ""

        tri_valid = safe_float(rowtri.get("valid_all"), None)
        tri_disp = safe_float(rowtri.get("disp_p95"), None)
        tri_z_neg = safe_float(rowtri.get("z_neg_ratio"), None)
        tri_z95_l1 = safe_float(rowtri.get("z_p95_L1"), None)
        tri_z95_l2 = safe_float(rowtri.get("z_p95_L2"), None)
        tri_z95_r1 = safe_float(rowtri.get("z_p95_R1"), None)
        tri_z95_r2 = safe_float(rowtri.get("z_p95_R2"), None)

        mon_valid = safe_float(rowmon.get("valid_all"), None)
        mon_disp = safe_float(rowmon.get("disp_p95"), None)
        mon_z95_l1 = safe_float(rowmon.get("z_p95_L1"), None)
        mon_z95_l2 = safe_float(rowmon.get("z_p95_L2"), None)
        mon_z95_r1 = safe_float(rowmon.get("z_p95_R1"), None)
        mon_z95_r2 = safe_float(rowmon.get("z_p95_R2"), None)

        gap_disp = (tri_disp - mon_disp) if (show_mon_metric and tri_disp is not None and mon_disp is not None) else None

        method_map = {"2d": "2d", "tri": "3d_tri", "monster": "3d_monster"}
        krow = kal_case_map.get((cid, method_map.get(chosen_code, "")), {}) if chosen_code != "None" else {}

        out = {
            "case_id": cid,
            "policy_mode": mode,
            "is_2d_ok": 1 if ok2d else 0,
            "reason_2d": reason2d,
            "2d_f1": fmt_num(safe_float(row2d.get("f1"), None)),
            "2d_mAP50": fmt_num(safe_float(row2d.get("mAP50"), None)),
            "is_tri_ok": (1 if oktri else 0) if show_tri_gate else hidden_tri,
            "reason_tri": reasontri if show_tri_gate else hidden_tri,
            "tri_valid_all": fmt_num_or_none(tri_valid) if show_tri_metric else hidden_tri,
            "tri_disp_p95": fmt_num_or_none(tri_disp) if show_tri_metric else hidden_tri,
            "tri_z_neg_ratio": fmt_num_or_none(tri_z_neg) if show_tri_metric else hidden_tri,
            "tri_z_p95_L1": fmt_num_or_none(tri_z95_l1) if show_tri_metric else hidden_tri,
            "tri_z_p95_L2": fmt_num_or_none(tri_z95_l2) if show_tri_metric else hidden_tri,
            "tri_z_p95_R1": fmt_num_or_none(tri_z95_r1) if show_tri_metric else hidden_tri,
            "tri_z_p95_R2": fmt_num_or_none(tri_z95_r2) if show_tri_metric else hidden_tri,
            "is_monster_ok": (1 if okmon else 0) if show_mon_gate else hidden_mon,
            "reason_monster": reasonmon if show_mon_gate else hidden_mon,
            "monster_valid_all": fmt_num_or_none(mon_valid) if show_mon_metric else hidden_mon,
            "monster_disp_p95": fmt_num_or_none(mon_disp) if show_mon_metric else hidden_mon,
            "monster_z_p95_L1": fmt_num_or_none(mon_z95_l1) if show_mon_metric else hidden_mon,
            "monster_z_p95_L2": fmt_num_or_none(mon_z95_l2) if show_mon_metric else hidden_mon,
            "monster_z_p95_R1": fmt_num_or_none(mon_z95_r1) if show_mon_metric else hidden_mon,
            "monster_z_p95_R2": fmt_num_or_none(mon_z95_r2) if show_mon_metric else hidden_mon,
            "_tri_monster_gap_disp_p95": fmt_num_or_none(gap_disp) if show_mon_metric else hidden_mon,
            "chosen_code": chosen_code,
            "chosen_disp_p95_raw": fmt_num(safe_float(krow.get("disp_p95_raw"), None)) if chosen_code != "None" else "None",
            "chosen_disp_p95_kf": fmt_num(safe_float(krow.get("disp_p95_kf"), None)) if chosen_code != "None" else "None",
            "chosen_kf_improve_disp_p95": fmt_num(safe_float(krow.get("improve_disp_p95"), None)) if chosen_code != "None" else "None",
            "chosen_jitter_med_raw": fmt_num(safe_float(krow.get("jitter_med_raw"), None)) if chosen_code != "None" else "None",
            "chosen_jitter_med_kf": fmt_num(safe_float(krow.get("jitter_med_kf"), None)) if chosen_code != "None" else "None",
            "chosen_kf_improve_jitter_med": fmt_num(safe_float(krow.get("improve_jitter_med"), None)) if chosen_code != "None" else "None",
        }
        rows.append(out)

    return rows, policy_obj


# pick representative cases
# ================================================
def pick_representative_cases(case_rows, bad_case_ids):
    rows = list(case_rows or [])

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
        s = str(x or "").strip().lower()
        if s in ("tri", "monster", "2d", "none"):
            return "None" if s == "none" else s
        return "None"

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
            return max(vals) - 500.0
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
            return max(vals) - 400.0
        return 0.0

    c1_2d_fail_none = []
    c2_fallback_2d = []
    c3_tri_pass = []
    c4_monster_rescue = []

    for r in rows:
        cid = str(r.get("case_id", "")).strip()
        if cid == "":
            continue

        tri_ok = parse_flag(r.get("is_tri_ok", ""))
        mon_ok = parse_flag(r.get("is_monster_ok", ""))
        ok2d = parse_flag(r.get("is_2d_ok", ""))
        mode = normalize_mode(r.get("policy_mode", ""))
        if mode == "None":
            mode2 = normalize_mode(r.get("chosen_code", ""))
            if mode2 != "None":
                mode = mode2

        if ok2d == 0 or mode == "None":
            c1_2d_fail_none.append(r)
            continue
        if mode == "2d" and tri_ok == 0 and mon_ok == 0:
            c2_fallback_2d.append(r)
            continue
        if mode == "tri" and tri_ok == 1:
            c3_tri_pass.append(r)
            continue
        if mode == "monster" and tri_ok == 0 and mon_ok == 1:
            c4_monster_rescue.append(r)
            continue

    def score_c1(r):
        f1 = to_float(r.get("2d_f1"))
        m50 = to_float(r.get("2d_mAP50"))
        return (-f1, -m50)

    def score_c2(r):
        tri_disp = to_float(r.get("tri_disp_p95"))
        mon_disp = to_float(r.get("monster_disp_p95"))
        fail_tri = tri_fail_margin(r)
        fail_mon = monster_fail_margin(r)
        vals = [v for v in (tri_disp, mon_disp) if np.isfinite(v)]
        max_disp = max(vals) if len(vals) > 0 else -1e9
        return (fail_mon, fail_tri, max_disp)

    def score_c3(r):
        tri_disp = to_float(r.get("tri_disp_p95"))
        tri_valid = to_float(r.get("tri_valid_all"))
        if not np.isfinite(tri_valid):
            tri_valid = 0.0
        if not np.isfinite(tri_disp):
            tri_disp = 1e6
        score = tri_valid - 0.001 * tri_disp
        return (score, tri_valid, -tri_disp)

    def score_c4(r):
        tri_disp = to_float(r.get("tri_disp_p95"))
        mon_disp = to_float(r.get("monster_disp_p95"))
        tri_valid = to_float(r.get("tri_valid_all"))
        mon_valid = to_float(r.get("monster_valid_all"))
        gap_disp = tri_disp - mon_disp if np.isfinite(tri_disp) and np.isfinite(mon_disp) else -1e9
        gap_valid = mon_valid - tri_valid if np.isfinite(mon_valid) and np.isfinite(tri_valid) else -1e9
        fail = tri_fail_margin(r)
        return (fail, gap_disp, gap_valid)

    c1_2d_fail_none = sorted(c1_2d_fail_none, key=score_c1, reverse=True)
    c2_fallback_2d = sorted(c2_fallback_2d, key=score_c2, reverse=True)
    c3_tri_pass = sorted(c3_tri_pass, key=score_c3, reverse=True)
    c4_monster_rescue = sorted(c4_monster_rescue, key=score_c4, reverse=True)

    def brief(r):
        if not r:
            return {}
        return {
            "case_id": str(r.get("case_id", "")),
            "policy_mode": r.get("policy_mode", ""),
            "chosen_code": r.get("chosen_code", ""),
            "reason_2d": r.get("reason_2d", ""),
            "reason_tri": r.get("reason_tri", ""),
            "reason_monster": r.get("reason_monster", ""),
            "2d_f1": r.get("2d_f1", ""),
            "tri_valid_all": r.get("tri_valid_all", ""),
            "tri_disp_p95": r.get("tri_disp_p95", ""),
            "monster_disp_p95": r.get("monster_disp_p95", ""),
            "chosen_kf_improve_disp_p95": r.get("chosen_kf_improve_disp_p95", ""),
            "chosen_kf_improve_jitter_med": r.get("chosen_kf_improve_jitter_med", ""),
        }

    return {
        "c1_2d_fail_none": brief(c1_2d_fail_none[0] if c1_2d_fail_none else None),
        "c2_fallback_2d": brief(c2_fallback_2d[0] if c2_fallback_2d else None),
        "c3_tri_pass": brief(c3_tri_pass[0] if c3_tri_pass else None),
        "c4_monster_rescue": brief(c4_monster_rescue[0] if c4_monster_rescue else None),
        "bad_case": {"case_id": bad_case_ids[0]} if bad_case_ids else {},
    }


# build run summary
# ================================================
def build_run_summary(policy_counts, bad_case_ids, model_row, test2d_all, tri_all, mon_all, kal_by_method):
    def km(method, key):
        row = kal_by_method.get(method, {})
        return row.get(key, "")

    model_pass = str(model_row.get("model_pass", "0")).strip().lower() in ("1", "true", "yes")

    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "total_cases": policy_counts["tri"] + policy_counts["monster"] + policy_counts["2d"] + policy_counts["None"],
        "policy_tri_cases": policy_counts["tri"],
        "policy_monster_cases": policy_counts["monster"],
        "policy_2d_cases": policy_counts["2d"],
        "policy_none_cases": policy_counts["None"],
        "bad_case_count": len(bad_case_ids),
        "bad_case_ids": "|".join(bad_case_ids),
        "bad_case_note": "bad case: visual model failed, kalman skipped in policy mode",
        "model_gate_pass": 1 if model_pass else 0,
        "model_gate_reason": model_row.get("reason", ""),
        "model_precision": fmt_num(safe_float(model_row.get("precision"), None)),
        "model_recall": fmt_num(safe_float(model_row.get("recall"), None)),
        "model_f1": fmt_num(safe_float(model_row.get("f1"), None)),
        "model_mAP50": fmt_num(safe_float(model_row.get("mAP50"), None)),
        "model_mAP50_95": fmt_num(safe_float(model_row.get("mAP50-95"), None)),
        "test2d_precision": fmt_num(safe_float(test2d_all.get("precision"), None)),
        "test2d_recall": fmt_num(safe_float(test2d_all.get("recall"), None)),
        "test2d_f1": fmt_num(safe_float(test2d_all.get("f1"), None)),
        "test2d_mAP50": fmt_num(safe_float(test2d_all.get("mAP50"), None)),
        "test2d_mAP50_95": fmt_num(safe_float(test2d_all.get("mAP50-95"), None)),
        "tri_valid_all": fmt_num(safe_float(tri_all.get("valid_all"), None)),
        "tri_disp_p95": fmt_num(safe_float(tri_all.get("disp_p95"), None)),
        "monster_valid_all": fmt_num(safe_float(mon_all.get("valid_all"), None)),
        "monster_disp_p95": fmt_num(safe_float(mon_all.get("disp_p95"), None)),
        "2d_disp_p95_raw": km("2d", "disp_p95_raw"),
        "2d_disp_p95_kf": km("2d", "disp_p95_kf"),
        "2d_improve_disp_p95": km("2d", "improve_disp_p95"),
        "tri_disp_p95_raw": km("3d_tri", "disp_p95_raw"),
        "tri_disp_p95_kf": km("3d_tri", "disp_p95_kf"),
        "tri_improve_disp_p95": km("3d_tri", "improve_disp_p95"),
        "monster_disp_p95_raw": km("3d_monster", "disp_p95_raw"),
        "monster_disp_p95_kf": km("3d_monster", "disp_p95_kf"),
        "monster_improve_disp_p95": km("3d_monster", "improve_disp_p95"),
    }


# save csv
# ================================================
def save_csv(rows, out_csv_path, cols):
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[DONE] saved: {out_csv_path}")


# save llm json
# ================================================
def save_llm_input(summary_row, policy_counts, bad_case_ids, kal_ablation_rows, model_row, policy_obj, case_rows, eval_dir):
    model_pass = str(model_row.get("model_pass", "0")).strip().lower() in ("1", "true", "yes")

    payload = {
        "summary": summary_row,
        "policy_counts": policy_counts,
        "bad_cases": {
            "count": len(bad_case_ids),
            "case_ids": bad_case_ids,
            "note": "mode=None means bad case, keep empty in policy-based output",
        },
        "kalman_ablation": kal_ablation_rows,
        "model_gate": {
            "pass": model_pass,
            "reason": model_row.get("reason", ""),
            "metrics": {
                "precision": safe_float(model_row.get("precision"), None),
                "recall": safe_float(model_row.get("recall"), None),
                "f1": safe_float(model_row.get("f1"), None),
                "mAP50": safe_float(model_row.get("mAP50"), None),
                "mAP50_95": safe_float(model_row.get("mAP50-95"), None),
            },
        },
        "thresholds": {
            "2d_gate": policy_obj.TH_2D,
            "tri_gate": policy_obj.TH_TRI,
            "monster_gate": policy_obj.TH_MONSTER,
        },
        "artifacts": {
            "final_case_table_csv": str((eval_dir / "final_case_table.csv").resolve()),
            "evaluate_kalman_csv": str((eval_dir / "evaluate_kalman.csv").resolve()),
        },
        "representative_cases": pick_representative_cases(case_rows, bad_case_ids),
    }

    out_path = eval_dir / "llm_input.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[DONE] saved: {out_path}")


# main
# ================================================
def main():
    eval_dir = EVAL_DIR

    # 0: load inputs
    policy_by_case, policy_counts, bad_case_ids = load_policy(eval_dir / "policy.csv")
    eval2d_by_case, eval2d_all = load_eval_by_case_and_all(eval_dir / "evaluate_2d_results.csv")
    tri_by_case, tri_all = load_eval_by_case_and_all(eval_dir / "evaluate_3d_results_tri.csv")
    mon_by_case, mon_all = load_eval_by_case_and_all(eval_dir / "evaluate_3d_results_monster.csv")
    kal_rows, kal_case_map = load_kalman_case(eval_dir / "evaluate_kalman.csv")
    kal_by_method, kal_ablation_rows = aggregate_kalman_by_method(kal_rows)
    model_row = load_model_gate(eval_dir / "evaluate_yolo_model.csv")

    # Step 1: final case table
    print("\n", "-" * 5, "Step 1: Build final case table", "-" * 5)
    case_rows, policy_obj = build_final_case_table(policy_by_case, eval2d_by_case, tri_by_case, mon_by_case, kal_case_map)
    case_cols = [
        "case_id", "policy_mode",
        "is_2d_ok", "reason_2d", "2d_f1", "2d_mAP50",
        "is_tri_ok", "reason_tri", "tri_valid_all", "tri_disp_p95", "tri_z_neg_ratio",
        "tri_z_p95_L1", "tri_z_p95_L2", "tri_z_p95_R1", "tri_z_p95_R2",
        "is_monster_ok", "reason_monster", "monster_valid_all", "monster_disp_p95",
        "monster_z_p95_L1", "monster_z_p95_L2", "monster_z_p95_R1", "monster_z_p95_R2",
        "chosen_code",
        "chosen_disp_p95_raw", "chosen_disp_p95_kf", "chosen_kf_improve_disp_p95",
        "chosen_jitter_med_raw", "chosen_jitter_med_kf", "chosen_kf_improve_jitter_med",
    ]
    save_csv(case_rows, eval_dir / "final_case_table.csv", case_cols)

    # Step 2: final summary
    print("\n", "-" * 5, "Step 2: Build final run summary", "-" * 5)
    summary_row = build_run_summary(policy_counts, bad_case_ids, model_row, eval2d_all, tri_all, mon_all, kal_by_method)
    save_csv([summary_row], eval_dir / "final_eval_summary.csv", list(summary_row.keys()))

    # Step 3: llm input
    print("\n", "-" * 5, "Step 3: Build llm input json", "-" * 5)
    save_llm_input(summary_row, policy_counts, bad_case_ids, kal_ablation_rows, model_row, policy_obj, case_rows, eval_dir)


if __name__ == "__main__":
    main()
