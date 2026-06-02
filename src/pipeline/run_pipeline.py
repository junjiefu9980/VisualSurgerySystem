"""
Run full visual pipeline.
"""
import csv
import sys
import tempfile
from datetime import datetime
from pathlib import Path


# ===============================================
# config
# ================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET_DIR = REPO_ROOT / "data" / "dataset"
YOLO_TEST_IMAGE_DIR = REPO_ROOT / "data" / "yolo_data" / "images" / "test"

DET_DIR = REPO_ROOT / "output" / "detections"
EVAL_DIR = REPO_ROOT / "output" / "evaluation"

CSV_2D_RESULTS = DET_DIR / "2d_results.csv"
CSV_3D_TRI_RESULTS = DET_DIR / "3d_tri_results.csv"
CSV_3D_MONSTER_RESULTS = DET_DIR / "3d_monster_results.csv"

CSV_2D_KALMAN_RESULTS = DET_DIR / "2d_kalman_results.csv"
CSV_3D_TRI_KALMAN_RESULTS = DET_DIR / "3d_tri_kalman_results.csv"
CSV_3D_MONSTER_KALMAN_RESULTS = DET_DIR / "3d_monster_kalman_results.csv"

EVAL_2D_RESULTS = EVAL_DIR / "evaluate_2d_results.csv"
EVAL_3D_RESULTS_TRI = EVAL_DIR / "evaluate_3d_results_tri.csv"
EVAL_3D_RESULTS_MONSTER = EVAL_DIR / "evaluate_3d_results_monster.csv"
EVAL_YOLO_MODEL = EVAL_DIR / "evaluate_yolo_model.csv"
POLICY_CSV = EVAL_DIR / "policy.csv"

# switches
SWITCH_MONSTER_TARGETED_REBUILD = False
SWITCH_SKIP_MODEL_EVAL = False

# load eval csv
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


# load policy cases
# ================================================
def load_policy_case_ids(policy_csv_path):
    policy_csv_path = Path(policy_csv_path)
    ids = []
    if not policy_csv_path.exists():
        return ids

    with open(policy_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = str(row.get("case_id", "")).strip()
            if case_id.isdigit():
                ids.append(case_id.zfill(6))
    return sorted(set(ids))


# load policy mode by case
# ================================================
def load_policy_mode_by_case(policy_csv_path):
    policy_csv_path = Path(policy_csv_path)
    data = {}
    if not policy_csv_path.exists():
        return data

    with open(policy_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = str(row.get("case_id", "")).strip()
            if not case_id.isdigit():
                continue
            mode = str(row.get("mode", "")).strip().lower()
            if mode not in ("2d", "tri", "monster"):
                mode = "None"
            data[case_id.zfill(6)] = mode
    return data


# temp case file
# ================================================
def write_temp_case_list(case_ids):
    case_ids = sorted({str(c).zfill(6) for c in case_ids if str(c).isdigit()})

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix="_case_list.csv", delete=False)
    with tmp:
        tmp.write("case_id\n")
        for cid in case_ids:
            tmp.write(f"{cid}\n")
    return Path(tmp.name)


def build_2d_pass_case_set(eval_2d_csv, policy_obj):
    eval_2d = load_eval_by_case(eval_2d_csv)
    case_2d_pass = set()
    for cid, row in eval_2d.items():
        if policy_obj.evaluate_2d(row):
            case_2d_pass.add(cid)
    return case_2d_pass


# model pass
# ================================================
def check_model_pass(model_eval_csv):
    model_eval_csv = Path(model_eval_csv)
    if not model_eval_csv.exists():
        return False

    with open(model_eval_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]

    if not rows:
        return False

    pass_val = str(rows[-1].get("model_pass", "0")).strip().lower()
    return pass_val in ("1", "true", "yes")


# route case ids
# ================================================
def build_route_case_sets(eval_2d_csv, eval_tri_csv, policy_obj):
    case_2d_pass = build_2d_pass_case_set(eval_2d_csv, policy_obj)
    eval_tri = load_eval_by_case(eval_tri_csv)

    case_tri_fail = set()
    for cid in case_2d_pass:
        tri_row = eval_tri.get(cid)
        if tri_row is None:
            case_tri_fail.add(cid)
            continue
        if not policy_obj.evaluate_tri(tri_row):
            case_tri_fail.add(cid)

    return case_2d_pass, case_tri_fail


# run kalman by mode
# ================================================
def run_kalman_by_mode(case_ids, mode, out_csv_path, data_2d, headers_2d, data_tri, headers_tri, data_mon, headers_mon):
    from src.pipeline import kalman_filter

    print("[INFO]", f"Kalman ({mode}) cases={len(case_ids)}")
    policy_by_case = {int(cid): mode for cid in case_ids}
    out_rows = kalman_filter.build_kalman_rows(
        policy_by_case,
        data_2d,
        data_tri,
        data_mon,
        verbose=False,
        only_policy_cases=True,
        strict_source_mode=mode,
        skip_bad_cases=True,
    )
    base_headers = kalman_filter.merge_headers(headers_2d, headers_tri, headers_mon)
    kalman_filter.save_csv(out_rows, base_headers, out_csv_path)


# main
# ================================================
def main():
    from src.yolo import yolo_detect
    from src.yolo import evaluate_yolo_model
    from src.pipeline import evaluate_2d_results
    from src.pipeline import build_3d_tri
    from src.pipeline import evaluate_3d_results
    from src.pipeline import build_3d_monster
    from src.pipeline import policy
    from src.pipeline import evaluate_kalman
    from src.pipeline import build_final_summary

    # 1. yolo 2d
    if YOLO_TEST_IMAGE_DIR.exists():
        print("\n", "=" * 20, "[1. YOLO 2D]", "=" * 20)
        yolo_detect.run_test_detection(
            model_path=None,
            image_dir=YOLO_TEST_IMAGE_DIR,
            out_csv=CSV_2D_RESULTS,
            conf=yolo_detect.YOLO_CONF_THR,
            iou=yolo_detect.YOLO_IOU_THR,
        )
    else:
        print("\n", "=" * 20, "[1. Reuse 2D results]", "=" * 20)
        print(f"[SKIP] test image dir not found: {YOLO_TEST_IMAGE_DIR}")
        print(f"[INFO] reuse: {CSV_2D_RESULTS}")

    if not CSV_2D_RESULTS.exists():
        print(f"[ERROR] missing 2D output: {CSV_2D_RESULTS}")
        raise SystemExit(1)

    # 2. evaluate 2d
    print("\n", "=" * 20, "[2. Evaluate 2D]", "=" * 20)
    gt_by_case = evaluate_2d_results.load_gt_by_case(evaluate_2d_results.GT_CSV_PATH)
    pred_by_case = evaluate_2d_results.load_pred_by_case(CSV_2D_RESULTS)
    rows, overall = evaluate_2d_results.evaluate_by_case(
        gt_by_case,
        pred_by_case,
        conf_thr=evaluate_2d_results.CONF_THR,
        iou_thr=evaluate_2d_results.IOU_THR,
    )
    evaluate_2d_results.save_eval_csv(rows, overall, EVAL_2D_RESULTS)

    # 3. evaluate yolo model
    if not SWITCH_SKIP_MODEL_EVAL:
        print("\n", "=" * 20, "[3. Evaluate YOLO model]", "=" * 20)
        all_row = evaluate_yolo_model.load_all_row(EVAL_2D_RESULTS)
        model_pass, reason = evaluate_yolo_model.is_model_pass(all_row)

        precision = ""
        recall = ""
        f1 = ""
        map50 = ""
        map5095 = ""
        if all_row is not None:
            precision = all_row.get("precision", "")
            recall = all_row.get("recall", "")
            f1 = all_row.get("f1", "")
            map50 = all_row.get("mAP50", "")
            map5095 = all_row.get("mAP50-95", "")

        row = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "eval_2d_csv": str(EVAL_2D_RESULTS),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mAP50": map50,
            "mAP50-95": map5095,
            "model_pass": "1" if model_pass else "0",
            "reason": reason,
        }
        evaluate_yolo_model.save_eval_row(row, EVAL_YOLO_MODEL)

        if not check_model_pass(EVAL_YOLO_MODEL):
            print("[ERROR] model-level gate failed, stop pipeline")
            raise SystemExit(2)
    else:
        print("\n", "=" * 20, "[3. Skip YOLO model eval]", "=" * 20)

    gate = policy.GatingPolicy()
    all_case_ids = sorted(load_eval_by_case(EVAL_2D_RESULTS).keys(), key=lambda x: int(x))

    # 4. build tri
    print("\n", "=" * 20, "[4. Build TRI 3D]", "=" * 20)
    case_2d_pass = build_2d_pass_case_set(EVAL_2D_RESULTS, gate)
    print(f"[INFO] tri target cases (2d pass): {len(case_2d_pass)}")
    tri_case_file = write_temp_case_list(case_2d_pass)
    try:
        build_3d_tri.run(CSV_2D_RESULTS, DATASET_DIR, CSV_3D_TRI_RESULTS, case_list_path=tri_case_file)
    finally:
        if tri_case_file.exists():
            tri_case_file.unlink()

    # 5. evaluate tri
    print("\n", "=" * 20, "[5. Evaluate TRI 3D]", "=" * 20)
    evaluate_3d_results.run(
        CSV_3D_TRI_RESULTS,
        EVAL_3D_RESULTS_TRI,
        case_filter=case_2d_pass,
        all_case_ids=all_case_ids,
    )

    # 6. build monster
    case_2d_pass, case_tri_fail = build_route_case_sets(EVAL_2D_RESULTS, EVAL_3D_RESULTS_TRI, gate)
    if SWITCH_MONSTER_TARGETED_REBUILD:
        print("\n", "=" * 20, "[6. Build MONSTER 3D (targeted)]", "=" * 20)
        print(f"[INFO] targeted cases: {len(case_tri_fail)}")

        tmp_case_file = write_temp_case_list(case_tri_fail)
        try:
            print("\n", "=" * 20, "[6. Build MONSTER 3D]", "=" * 20)
            build_3d_monster.run(
                CSV_2D_RESULTS,
                DATASET_DIR,
                CSV_3D_MONSTER_RESULTS,
                YOLO_TEST_IMAGE_DIR,
                case_list_path=tmp_case_file,
            )
        finally:
            if tmp_case_file.exists():
                tmp_case_file.unlink()
    else:
        print("\n", "=" * 20, "[6. Reuse MONSTER 3D local file]", "=" * 20)
        if not CSV_3D_MONSTER_RESULTS.exists():
            print(f"[ERROR] monster csv missing: {CSV_3D_MONSTER_RESULTS}")
            raise SystemExit(1)
        print(f"[INFO] reuse: {CSV_3D_MONSTER_RESULTS}")

    # 7. evaluate monster
    print("\n", "=" * 20, "[7. Evaluate MONSTER 3D]", "=" * 20)
    evaluate_3d_results.run(
        CSV_3D_MONSTER_RESULTS,
        EVAL_3D_RESULTS_MONSTER,
        case_filter=case_tri_fail,
        all_case_ids=all_case_ids,
    )

    # 8. policy
    print("\n", "=" * 20, "[8. Policy]", "=" * 20)
    policy.main()

    # 9. kalman results
    policy_mode_by_case = load_policy_mode_by_case(POLICY_CSV)
    case_ids_2d = sorted([cid for cid, mode in policy_mode_by_case.items() if mode in ("2d", "tri", "monster")], key=lambda x: int(x))
    case_ids_tri = sorted([cid for cid, mode in policy_mode_by_case.items() if mode in ("tri", "monster")], key=lambda x: int(x))
    case_ids_mon = sorted([cid for cid, mode in policy_mode_by_case.items() if mode == "monster"], key=lambda x: int(x))

    print("\n", "=" * 20, "[9. Build method Kalman outputs]", "=" * 20)
    from src.pipeline import kalman_filter
    data_2d, headers_2d = kalman_filter.load_detection_by_case(CSV_2D_RESULTS, tag="2D", verbose=False)
    data_tri, headers_tri = kalman_filter.load_detection_by_case(CSV_3D_TRI_RESULTS, tag="TRI-3D", verbose=False)
    data_mon, headers_mon = kalman_filter.load_detection_by_case(CSV_3D_MONSTER_RESULTS, tag="MONSTER-3D", verbose=False)
    run_kalman_by_mode(case_ids_2d, "2d", CSV_2D_KALMAN_RESULTS, data_2d, headers_2d, data_tri, headers_tri, data_mon, headers_mon)
    run_kalman_by_mode(case_ids_tri, "tri", CSV_3D_TRI_KALMAN_RESULTS, data_2d, headers_2d, data_tri, headers_tri, data_mon, headers_mon)
    run_kalman_by_mode(case_ids_mon, "monster", CSV_3D_MONSTER_KALMAN_RESULTS, data_2d, headers_2d, data_tri, headers_tri, data_mon, headers_mon)

    # 10. evaluate kalman
    print("\n", "=" * 20, "[10. Evaluate Kalman]", "=" * 20)
    evaluate_kalman.main()

    # 11. final summary
    print("\n", "=" * 20, "[11. Build Final Summary]", "=" * 20)
    build_final_summary.main()


    print("\n", "=" * 10,"[DONE]", "All the visual pipeline finished", "=" * 10)


if __name__ == "__main__":
    main()
