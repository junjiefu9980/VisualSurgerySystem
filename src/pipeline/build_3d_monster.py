"""
Step B2: build monster 3D from 2d_results.csv.
"""
import copy
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import InterpolationMode, Resize


# ===============================================
# config
# ================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
DET_DIR = REPO_ROOT / "output" / "detections"
EVAL_DIR = REPO_ROOT / "output" / "evaluation"
DATASET_DIR = REPO_ROOT / "data" / "dataset"
YOLO_TEST_IMAGE_DIR = REPO_ROOT / "data" / "yolo_data" / "images" / "test"
MONSTER_DIR = REPO_ROOT / "monster"

IN_CSV_PATH = DET_DIR / "2d_results.csv"
CALIB_ROOT = DATASET_DIR
IMAGE_ROOT = YOLO_TEST_IMAGE_DIR
OUT_CSV_PATH = DET_DIR / "3d_monster_results.csv"
TRI_CSV_PATH = DET_DIR / "3d_tri_results.csv"
EVAL_2D_RESULTS_PATH = EVAL_DIR / "evaluate_2d_results.csv"
EVAL_3D_TRI_RESULTS_PATH = EVAL_DIR / "evaluate_3d_results_tri.csv"

# image size, keep same as monster script
IMG_ORIG_W, IMG_ORIG_H = 1400, 986
IMG_NEW_W, IMG_NEW_H = 640, 512
SCALE_X = IMG_NEW_W / IMG_ORIG_W
SCALE_Y = IMG_NEW_H / IMG_ORIG_H

KPT_LABELS = ["L1", "L2", "R1", "R2"]

try:
    from src.pipeline.policy import GatingPolicy
except Exception:
    GatingPolicy = None


# ===============================================
# import monster modules
# ================================================
if MONSTER_DIR.exists() and str(MONSTER_DIR) not in sys.path:
    sys.path.insert(0, str(MONSTER_DIR))
if (MONSTER_DIR / "MonSter").exists() and str(MONSTER_DIR / "MonSter") not in sys.path:
    sys.path.insert(0, str(MONSTER_DIR / "MonSter"))

_IMPORT_ERR_RECT = ""
_IMPORT_ERR_DEPTH = ""

try:
    from stereo_rectify import StereoRectifier
except Exception as e:
    StereoRectifier = None
    _IMPORT_ERR_RECT = str(e)

try:
    from depth_estimator_monster import DepthEstimator, MonSter_config
except Exception as e:
    DepthEstimator = None
    MonSter_config = None
    _IMPORT_ERR_DEPTH = str(e)


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


# load eval by case
# ================================================
def load_eval_by_case(csv_path):
    csv_path = Path(csv_path)
    data = {}
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


# route from eval
# ================================================
def build_case_filter_from_eval(eval_2d_path, eval_tri_path):
    if GatingPolicy is None:
        raise RuntimeError("policy import failed, cannot build monster gate by policy functions")

    missing = []
    eval_2d_path = Path(eval_2d_path)
    eval_tri_path = Path(eval_tri_path)
    if not eval_2d_path.exists():
        missing.append(str(eval_2d_path))
    if not eval_tri_path.exists():
        missing.append(str(eval_tri_path))
    if len(missing) > 0:
        raise FileNotFoundError("missing eval file(s):\n- " + "\n- ".join(missing))

    policy_obj = GatingPolicy()
    eval_2d = load_eval_by_case(eval_2d_path)
    eval_tri = load_eval_by_case(eval_tri_path)

    case_ids = set()
    for cid, row2d in eval_2d.items():
        if not policy_obj.evaluate_2d(row2d):
            continue
        tri_row = eval_tri.get(cid)
        if tri_row is None:
            case_ids.add(cid)
            continue
        if not policy_obj.evaluate_tri(tri_row):
            case_ids.add(cid)

    # MONSTER is not the first route.
    # It is only prepared for cases that passed 2D but still failed TRI.
    print(f"[INFO] target cases from eval (2d pass + tri fail): {len(case_ids)}")
    return case_ids


# unproject
# ================================================
def unproject_point(u_d, v_d, depth, k_new):
    if depth <= 0 or not np.isfinite(depth):
        return np.nan, np.nan, np.nan

    fx = k_new[0, 0]
    fy = k_new[1, 1]
    cx = k_new[0, 2]
    cy = k_new[1, 2]

    x = (u_d - cx) * depth / fx
    y = (v_d - cy) * depth / fy
    z = depth
    return float(x), float(y), float(z)


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


# config init
# ================================================
def init_monster_config():
    missing_items = []
    if not (MONSTER_DIR / "stereo_rectify.py").exists():
        missing_items.append(str(MONSTER_DIR / "stereo_rectify.py"))
    if not (MONSTER_DIR / "depth_estimator_monster.py").exists():
        missing_items.append(str(MONSTER_DIR / "depth_estimator_monster.py"))
    if not (MONSTER_DIR / "MonSter").exists():
        missing_items.append(str(MONSTER_DIR / "MonSter"))
    if len(missing_items) > 0:
        raise FileNotFoundError("monster files missing:\n- " + "\n- ".join(missing_items))

    if StereoRectifier is None:
        msg = _IMPORT_ERR_RECT if _IMPORT_ERR_RECT != "" else "unknown"
        raise RuntimeError(f"stereo_rectify import failed: {msg}")
    if DepthEstimator is None or MonSter_config is None:
        msg = _IMPORT_ERR_DEPTH if _IMPORT_ERR_DEPTH != "" else "unknown"
        raise RuntimeError(f"depth_estimator_monster import failed: {msg}")

    pretrained_dir_a = MONSTER_DIR
    pretrained_dir_b = MONSTER_DIR / "MonSter" / "pretrained"
    vitl_path = pretrained_dir_a / "depth_anything_v2_vitl.pth"
    mix_path = pretrained_dir_a / "mix_all.pth"
    if not vitl_path.exists():
        vitl_path = pretrained_dir_b / "depth_anything_v2_vitl.pth"
    if not mix_path.exists():
        mix_path = pretrained_dir_b / "mix_all.pth"
    if not vitl_path.exists() or not mix_path.exists():
        raise FileNotFoundError(
            f"monster weights missing under: {MONSTER_DIR}\n"
            "need: depth_anything_v2_vitl.pth and mix_all.pth "
            "(in monster/ or monster/MonSter/pretrained/)"
        )

    # This local MONSTER path is the heavy branch.
    # The current script expects CUDA and pre-downloaded checkpoints.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. MonSter inference in current script requires CUDA.")

    cfg = copy.deepcopy(MonSter_config)
    cfg["pretrained_dir"] = str(vitl_path.parent)
    cfg["pretrained"] = str(mix_path)
    return cfg


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
    # Reuse 000000 as a safe local fallback for packed submissions.
    p3 = calib_root / "000000" / "StereoCalibrationDVRK.ini"
    if p3.exists():
        return p3
    return None


# save csv
# ================================================
def save_rows(fieldnames, rows, out_csv_path, extra_cols):
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    out_fields = list(fieldnames)
    for c in extra_cols:
        if c not in out_fields:
            out_fields.append(c)

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] saved: {out_csv_path}")
    print(f"[INFO] rows={len(rows)}")


# run core
# ================================================
def run(in_csv_path, calib_root, out_csv_path, image_root, case_list_path=None):
    # Step 1: load csv
    print("\n", "-" * 5, "Step 1: Loading 2D csv", "-" * 5)
    case_filter = load_case_filter(case_list_path)
    if case_filter is None:
        case_filter = build_case_filter_from_eval(EVAL_2D_RESULTS_PATH, EVAL_3D_TRI_RESULTS_PATH)
    fieldnames, rows = load_rows(in_csv_path)
    if not rows:
        print("[ERROR] no rows loaded")
        return None

    out_csv_path = Path(out_csv_path)
    if out_csv_path.resolve() == TRI_CSV_PATH.resolve():
        raise ValueError("monster output cannot overwrite tri output")

    # Step 2: prepare monster
    print("\n", "-" * 5, "Step 2: Preparing monster", "-" * 5)
    monster_cfg = init_monster_config()
    resize = Resize((IMG_NEW_H, IMG_NEW_W), interpolation=InterpolationMode.BILINEAR)

    pair_rows = defaultdict(dict)
    for row in rows:
        case_id, side, frame_idx = parse_filename(row.get("filename", ""))
        if case_id is None:
            continue
        pair_rows[(case_id, frame_idx)][side] = row

    extra_cols = [f"{label}_{a}3d" for label in KPT_LABELS for a in ["x", "y", "z"]]
    for row in rows:
        for c in extra_cols:
            row[c] = ""

    # Only keep valid stereo pairs from the target case list.
    # Cases outside the selected fallback set keep empty MONSTER fields.
    pair_keys = []
    for key, sides in pair_rows.items():
        case_id = key[0]
        if "left" not in sides or "right" not in sides:
            continue
        if case_filter is not None and case_id not in case_filter:
            continue
        pair_keys.append(key)
    pair_keys.sort(key=lambda x: (x[0], x[1]))
    print(f"[INFO] stereo pairs: {len(pair_keys)}")

    rect_cache = {}
    k_new_cache = {}

    current_case = None
    depth_estimator = None

    ok = 0
    skip = 0

    for case_id, frame_idx in pair_keys:
        left_row = pair_rows[(case_id, frame_idx)]["left"]
        right_row = pair_rows[(case_id, frame_idx)]["right"]

        left_img_path = Path(image_root) / f"{case_id}_left_video_{frame_idx}.jpg"
        right_img_path = Path(image_root) / f"{case_id}_right_video_{frame_idx}.jpg"
        ini_path = find_calib_ini_path(calib_root, case_id)

        if ini_path is None or not left_img_path.exists() or not right_img_path.exists():
            skip += 1
            continue

        if case_id not in rect_cache:
            rect = StereoRectifier(str(ini_path), img_size_new=None, mode="conventional")  # type: ignore
            calib = rect.get_rectified_calib()
            k_left = calib["intrinsics"]["left"].astype(np.float32)
            k_new = np.array([
                [k_left[0, 0] * SCALE_X, 0, k_left[0, 2] * SCALE_X],
                [0, k_left[1, 1] * SCALE_Y, k_left[1, 2] * SCALE_Y],
                [0, 0, 1],
            ])
            rect_cache[case_id] = rect
            k_new_cache[case_id] = k_new

        # Reload the depth estimator when case_id changes.
        # This keeps memory use more stable during long runs.
        if current_case != case_id:
            current_case = case_id
            depth_estimator = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if depth_estimator is None:
            print(f"[INFO] load monster model for case {case_id}")
            depth_estimator = DepthEstimator(monster_cfg, "monster")  # type: ignore

        rect = rect_cache[case_id]
        k_new = k_new_cache[case_id]

        left_bgr = cv2.imread(str(left_img_path))
        right_bgr = cv2.imread(str(right_img_path))
        if left_bgr is None or right_bgr is None:
            skip += 1
            continue

        left_img = torch.from_numpy(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        right_img = torch.from_numpy(cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()

        left_img, right_img = rect(left_img, right_img)
        left_img = resize(left_img)
        right_img = resize(right_img)

        baseline = np.atleast_1d(rect.get_rectified_calib()["bf"].astype(np.float32))
        scale = IMG_NEW_W / IMG_ORIG_W

        # MONSTER predicts depth on rectified and resized stereo views.
        with torch.no_grad():
            depth = depth_estimator(left_img[None], right_img[None], baseline[None] * scale)

        depth_np = np.squeeze(depth[0, 0].cpu().numpy())
        if depth_np.shape != (IMG_NEW_H, IMG_NEW_W):
            depth_np = cv2.resize(depth_np.astype(np.float32), (IMG_NEW_W, IMG_NEW_H), interpolation=cv2.INTER_LINEAR)

        h_d, w_d = depth_np.shape

        for label in KPT_LABELS:
            x_orig = safe_float(left_row.get(f"{label}_x"))
            y_orig = safe_float(left_row.get(f"{label}_y"))
            if np.isnan(x_orig) or np.isnan(y_orig):
                continue

            u_d = x_orig * SCALE_X
            v_d = y_orig * SCALE_Y
            u_i = int(round(np.clip(u_d, 0, w_d - 1)))
            v_i = int(round(np.clip(v_d, 0, h_d - 1)))
            d = float(depth_np[v_i, u_i])

            # Use the left image keypoint and the predicted depth map
            # to recover a project-space 3D point.
            x3, y3, z3 = unproject_point(u_d, v_d, d, k_new)
            val_x = "" if not np.isfinite(x3) else round(x3, 4)
            val_y = "" if not np.isfinite(y3) else round(y3, 4)
            val_z = "" if not np.isfinite(z3) else round(z3, 4)

            for row in (left_row, right_row):
                row[f"{label}_x3d"] = val_x
                row[f"{label}_y3d"] = val_y
                row[f"{label}_z3d"] = val_z

        ok += 1
        if ok % 200 == 0:
            print(f"[INFO] progress: ok={ok}, skip={skip}")

    # Step 3: save csv
    print("\n", "-" * 5, "Step 3: Saving 3D csv", "-" * 5)
    save_rows(fieldnames, rows, out_csv_path, extra_cols)

    print(f"[INFO] ok_pairs={ok}, skipped_pairs={skip}")
    return str(out_csv_path)


# main
# ================================================
def main():
    run(IN_CSV_PATH, CALIB_ROOT, OUT_CSV_PATH, IMAGE_ROOT, case_list_path=None)


if __name__ == "__main__":
    main()
