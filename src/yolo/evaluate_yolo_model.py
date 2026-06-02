# evaluate YOLO model conclusion from 2D eval csv
import csv
import os
from datetime import datetime

# set paths
GenMu = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PingGuMu = os.path.join(GenMu, "output", "evaluation")
Ev2dCsv = os.path.join(PingGuMu, "evaluate_2d_results.csv")
ShuChuCsv = os.path.join(PingGuMu, "evaluate_yolo_model.csv")

# set pass thresholds
YuZhiF1 = 0.85
YuZhiMp = 0.80


def AnQuFlt(ShuRu, MoRen=None):
    # safe float conversion
    try:
        if ShuRu is None or ShuRu == "":
            return MoRen
        return float(ShuRu)
    except (TypeError, ValueError):
        return MoRen


def DuAllHg(CsvLuJi):
    # load the ALL summary row from eval csv
    if not os.path.exists(CsvLuJi):
        return None
    HangLie = []
    with open(CsvLuJi, "r", encoding="utf-8") as fp:
        for HangGe in csv.DictReader(fp):
            HangLie.append(HangGe)
    for HangGe in HangLie:
        if str(HangGe.get("case_id", "")).strip().upper() == "ALL":
            return HangGe
    if HangLie:
        return HangLie[-1]
    return None

# check if model passes gate
def PanDuanTg(AllHang):
    if AllHang is None:
        return False, "missing_eval"
    F1Zhi = AnQuFlt(AllHang.get("f1"), 0.0)
    MpZhi = AnQuFlt(AllHang.get("mAP50"), 0.0)
    if F1Zhi < YuZhiF1:
        return False, f"f1<{YuZhiF1}"
    if MpZhi < YuZhiMp:
        return False, f"mAP50<{YuZhiMp}"
    return True, "pass"

# save evaluation result row
def CunPgJg(HangGe, ShuChuLj):
    os.makedirs(os.path.dirname(ShuChuLj), exist_ok=True)
    LieMing = ["time", "eval_2d_csv", "precision", "recall", "f1", "mAP50", "mAP50-95", "model_pass", "reason"]
    with open(ShuChuLj, "w", newline="", encoding="utf-8") as fp:
        XieRu = csv.DictWriter(fp, fieldnames=LieMing)
        XieRu.writeheader()
        XieRu.writerow(HangGe)
    print(f"[DONE] saved: {ShuChuLj}")

# export aliases
load_all_row = DuAllHg
is_model_pass = PanDuanTg
save_eval_row = CunPgJg


def main():
    print("\n", "-" * 5, "Step 1: Loading 2D eval", "-" * 5)
    AllHang = DuAllHg(Ev2dCsv)
    print("\n", "-" * 5, "Step 2: Model conclusion", "-" * 5)
    TongGuo, YuanYin = PanDuanTg(AllHang)
    JingDu = ""
    ZhaoHui = ""
    F1Zhi = ""
    MpZhi = ""
    Mp95 = ""
    if AllHang is not None:
        JingDu = AllHang.get("precision", "")
        ZhaoHui = AllHang.get("recall", "")
        F1Zhi = AllHang.get("f1", "")
        MpZhi = AllHang.get("mAP50", "")
        Mp95 = AllHang.get("mAP50-95", "")
    print(f"[INFO] f1={F1Zhi}, mAP50={MpZhi}")
    print(f"[INFO] pass={TongGuo} ({YuanYin})")
    print("\n", "-" * 5, "Step 3: Saving", "-" * 5)
    HangGe = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "eval_2d_csv": Ev2dCsv,
        "precision": JingDu, "recall": ZhaoHui, "f1": F1Zhi,
        "mAP50": MpZhi, "mAP50-95": Mp95,
        "model_pass": "1" if TongGuo else "0", "reason": YuanYin,
    }
    CunPgJg(HangGe, ShuChuCsv)
    if not TongGuo:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
