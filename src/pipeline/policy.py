import csv
from pathlib import Path


# ===============================================
# path and settings
# ================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "output" / "evaluation"

CSV_2D_PATH = EVAL_DIR / "evaluate_2d_results.csv"
CSV_TRI_PATH = EVAL_DIR / "evaluate_3d_results_tri.csv"
CSV_MONSTER_PATH = EVAL_DIR / "evaluate_3d_results_monster.csv"
OUT_CSV_PATH = EVAL_DIR / "policy.csv"


# ===============================================
# policy
# ================================================
class Mode:
    TRI = "tri"
    MONSTER = "monster"
    MODE_2D = "2d"
    NONE = "None"


class GatingPolicy:
    def __init__(self):
        # 2d gate
        self.TH_2D = {
            "f1_min": 0.85,
            "map50_min": 0.80,
        }

        # tri gate (slightly relaxed)
        self.TH_TRI = {
            "valid_all_min": 0.80,
            "z_p95_max": 300.0,
            "disp_p95_max": 260.0,
            "z_neg_ratio_max": 0.10,
        }

        # monster gate (keep strict)
        self.TH_MONSTER = {
            "valid_all_min": 0.85,
            "z_p95_max": 300.0,
            "disp_p95_max": 90.0,
        }

    def safe_float(self, x, default=0.0):
        try:
            if x is None or x == "":
                return default
            return float(x)
        except (TypeError, ValueError):
            return default

    def evaluate_2d(self, m2d):
        if self.safe_float(m2d.get("f1"), 0.0) < self.TH_2D["f1_min"]:
            return False
        if self.safe_float(m2d.get("mAP50"), 0.0) < self.TH_2D["map50_min"]:
            return False
        return True

    def evaluate_tri(self, mtri):
        if self.safe_float(mtri.get("valid_all"), 0.0) < self.TH_TRI["valid_all_min"]:
            return False
        if self.safe_float(mtri.get("disp_p95"), 9999.0) > self.TH_TRI["disp_p95_max"]:
            return False
        if self.safe_float(mtri.get("z_neg_ratio"), 1.0) > self.TH_TRI["z_neg_ratio_max"]:
            return False

        for k in ["z_p95_L1", "z_p95_L2", "z_p95_R1", "z_p95_R2"]:
            if self.safe_float(mtri.get(k), 9999.0) > self.TH_TRI["z_p95_max"]:
                return False

        return True

    def evaluate_monster(self, mmon):
        if self.safe_float(mmon.get("valid_all"), 0.0) < self.TH_MONSTER["valid_all_min"]:
            return False
        if self.safe_float(mmon.get("disp_p95"), 9999.0) > self.TH_MONSTER["disp_p95_max"]:
            return False

        for k in ["z_p95_L1", "z_p95_L2", "z_p95_R1", "z_p95_R2"]:
            if self.safe_float(mmon.get(k), 9999.0) > self.TH_MONSTER["z_p95_max"]:
                return False

        return True

    def decide_mode(self, m2d, mtri, mmon):
        # Gate A: 2d must pass first
        if not m2d or not self.evaluate_2d(m2d):
            return Mode.NONE

        # Gate B1: tri first
        if mtri and self.evaluate_tri(mtri):
            return Mode.TRI

        # Gate B2: monster fallback
        if mmon and self.evaluate_monster(mmon):
            return Mode.MONSTER

        # both 3d fail, fallback to 2d
        return Mode.MODE_2D


# ===============================================
# file
# ================================================
def load_csv_to_dict(csv_path):
    data = {}
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"[WARNING] file not found: {csv_path}")
        return data

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = str(row.get("case_id", "")).strip()
            if case_id == "" or case_id == "ALL":
                continue
            data[case_id] = row

    return data


def save_policy_csv(rows, out_csv_path):
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "mode"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] saved: {out_csv_path}")


# ===============================================
# main
# ================================================
def main():
    # load eval csv
    data_2d = load_csv_to_dict(CSV_2D_PATH)
    data_tri = load_csv_to_dict(CSV_TRI_PATH)
    data_mon = load_csv_to_dict(CSV_MONSTER_PATH)

    all_case_ids = set(data_2d.keys()) | set(data_tri.keys()) | set(data_mon.keys())
    if not all_case_ids:
        print("[ERROR] no case found")
        return

    # run policy
    policy = GatingPolicy()
    rows = []

    count = {Mode.TRI: 0, Mode.MONSTER: 0, Mode.MODE_2D: 0, Mode.NONE: 0}

    for case_id in sorted(all_case_ids, key=lambda x: int(x) if x.isdigit() else x):
        m2d = data_2d.get(case_id, {})
        mtri = data_tri.get(case_id, {})
        mmon = data_mon.get(case_id, {})

        mode = policy.decide_mode(m2d, mtri, mmon)
        rows.append({"case_id": case_id, "mode": mode})

        if mode in count:
            count[mode] += 1

    print(
        f"[INFO] mode count: tri={count[Mode.TRI]}, monster={count[Mode.MONSTER]}, "
        f"2d={count[Mode.MODE_2D]}, None={count[Mode.NONE]}"
    )

    # save csv
    save_policy_csv(rows, OUT_CSV_PATH)


if __name__ == "__main__":
    main()
