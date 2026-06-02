# use triangle or monster to evaluate 3D results

import csv
import math
import re
import os

import numpy as np


GenMl = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JcMl = os.path.join(GenMl, "output", "detections")
PgMl = os.path.join(GenMl, "output", "evaluation")

RuCsv = os.path.join(JcMl, "3d_tri_results.csv")
ScCsv = os.path.join(PgMl, "evaluate_3d_results_tri.csv")

GjDian = ["L1", "L2", "R1", "R2"]
ZbLie = [
    "valid_L1", "valid_L2", "valid_R1", "valid_R2", "valid_all",
    "z_neg_ratio",
    "z_med_L1", "z_med_L2", "z_med_R1", "z_med_R2",
    "z_p95_L1", "z_p95_L2", "z_p95_R1", "z_p95_R2",
    "disp_med", "disp_p95",
]


def AnQFlt(X, MoRen=None):
    try:
        if X is None:
            return MoRen
        V = float(X)
        if math.isnan(V):
            return MoRen
        return V
    except (TypeError, ValueError):
        return MoRen

# parse filename
def JxWjMing(WjMing):
    Jg = os.path.splitext(os.path.basename(str(WjMing)))[0]
    BuFen = Jg.split("_")
    if len(BuFen) < 3 or not re.fullmatch(r"\d{6}", BuFen[0]):
        return None
    Cid = int(BuFen[0])
    ZhHao = int(BuFen[-1])
    MingZi = "_".join(BuFen[1:-1])
    FangX = "left" if "left" in MingZi.lower() else "right"
    return Cid, FangX, ZhHao

# load 3D results (left rows only)
def DuSjSj(CsvLj, AlGuoLv=None):
    print("\n", "-" * 5, "Step 1: Loading 3D", "-" * 5)
    if not os.path.exists(CsvLj):
        print(f"[ERROR] {CsvLj} not exist")
        return {}

    SjMap = {}
    ZongSh = 0
    TiaoGo = 0

    with open(CsvLj, "r", encoding="utf-8") as f:
        for Hang in csv.DictReader(f):
            ZongSh += 1
            Jx = JxWjMing(Hang.get("filename", ""))
            if Jx is None:
                TiaoGo += 1
                continue
            Cid, FangX, ZhHao = Jx
            if FangX != "left":
                continue

            CsJian = str(Cid).zfill(6)
            if AlGuoLv is not None and CsJian not in AlGuoLv:
                continue

            GjD = {}
            for K in GjDian:
                Xv = AnQFlt(Hang.get(f"{K}_x3d"))
                Yv = AnQFlt(Hang.get(f"{K}_y3d"))
                Zv = AnQFlt(Hang.get(f"{K}_z3d"))
                GjD[K] = (Xv, Yv, Zv)

            if Cid not in SjMap:
                SjMap[Cid] = []
            SjMap[Cid].append((ZhHao, GjD))

    for Cid in SjMap:
        SjMap[Cid].sort(key=lambda x: x[0])

    NZh = sum(len(V) for V in SjMap.values())
    print(f"[DONE] cases={len(SjMap)}, left frames={NZh}")
    return SjMap

# evaluate one case
def PgYiAl(ZhLie):
    N = len(ZhLie)
    YxShu = {K: 0 for K in GjDian}
    FuShu = {K: 0 for K in GjDian}
    ZLie = {K: [] for K in GjDian}
    WyAll = []
    ShXyz = {K: None for K in GjDian}

    for _, GjD in ZhLie:
        for K in GjDian:
            Xv, Yv, Zv = GjD[K]
            if Xv is None or Yv is None or Zv is None:
                ShXyz[K] = None
                continue
            if not (np.isfinite(Xv) and np.isfinite(Yv) and np.isfinite(Zv)):
                ShXyz[K] = None
                continue
            if Zv <= 0:
                FuShu[K] += 1
                ShXyz[K] = None
                continue

            YxShu[K] += 1
            ZLie[K].append(Zv)

            if ShXyz[K] is not None:
                Dx = Xv - ShXyz[K][0]
                Dy = Yv - ShXyz[K][1]
                Dz = Zv - ShXyz[K][2]
                WyAll.append(math.sqrt(Dx * Dx + Dy * Dy + Dz * Dz))
            ShXyz[K] = (Xv, Yv, Zv)

    YxBi = {}
    for K in GjDian:
        YxBi[K] = YxShu[K] / N if N > 0 else 0.0
    YxAll = sum(YxBi.values()) / len(GjDian)

    ZongYx = sum(YxShu.values()) + sum(FuShu.values())
    FuBi = sum(FuShu.values()) / ZongYx if ZongYx > 0 else 0.0

    def ZhWei(Lst):
        return float(np.nanmedian(Lst)) if Lst else None

    def Bfwy(Lst):
        return float(np.nanpercentile(Lst, 95)) if Lst else None

    return {
        "valid_L1": YxBi["L1"], "valid_L2": YxBi["L2"],
        "valid_R1": YxBi["R1"], "valid_R2": YxBi["R2"],
        "valid_all": YxAll, "z_neg_ratio": FuBi,
        "z_med_L1": ZhWei(ZLie["L1"]), "z_med_L2": ZhWei(ZLie["L2"]),
        "z_med_R1": ZhWei(ZLie["R1"]), "z_med_R2": ZhWei(ZLie["R2"]),
        "z_p95_L1": Bfwy(ZLie["L1"]), "z_p95_L2": Bfwy(ZLie["L2"]),
        "z_p95_R1": Bfwy(ZLie["R1"]), "z_p95_R2": Bfwy(ZLie["R2"]),
        "disp_med": ZhWei(WyAll), "disp_p95": Bfwy(WyAll),
    }


def KongHang(Cid):
    Hang = {"case_id": str(Cid)}
    for C in ZbLie:
        Hang[C] = "None"
    return Hang


def CunCsv(HangLi, ZongJg, ScLj):
    print("\n", "-" * 5, "Step 3: Saving", "-" * 5)
    os.makedirs(os.path.dirname(ScLj), exist_ok=True)

    LieMing = ["case_id"] + ZbLie

    def GeShi(X):
        if X is None:
            return "None"
        if isinstance(X, str):
            return X
        try:
            return round(float(X), 6)
        except Exception:
            return ""

    with open(ScLj, "w", newline="", encoding="utf-8") as f:
        Xie = csv.writer(f)
        Xie.writerow(LieMing)
        for R in HangLi:
            Xie.writerow([R.get("case_id", "")] + [GeShi(R.get(C)) for C in ZbLie])
        Xie.writerow([ZongJg.get("case_id", "")] + [GeShi(ZongJg.get(C)) for C in ZbLie])
    print(f"[DONE] Saved: {ScLj}")


IN_3D_CSV_PATH = RuCsv
OUT_CSV_PATH = ScCsv
KPTS = GjDian
METRIC_COLS = ZbLie


def load_data(CsvLj, case_filter=None):
    return DuSjSj(CsvLj, AlGuoLv=case_filter)


def evaluate_case(ZhLie):
    return PgYiAl(ZhLie)


def save_csv(HangLi, ZongJg, ScLj):
    return CunCsv(HangLi, ZongJg, ScLj)


def build_none_case_row(Cid):
    return KongHang(Cid)


def run(RuLj, ScLj, case_filter=None, all_case_ids=None):
    SjMap = DuSjSj(RuLj, AlGuoLv=case_filter)

    print("\n", "-" * 5, "Step 2: Evaluating", "-" * 5)
    AlJg = {}
    SuoYZh = []

    for Cid in sorted(SjMap.keys()):
        ZhLie = SjMap[Cid]
        SuoYZh.extend(ZhLie)
        M = PgYiAl(ZhLie)
        CsJian = str(Cid).zfill(6)
        M["case_id"] = str(int(CsJian))
        AlJg[CsJian] = M

    HangLi = []
    if all_case_ids is not None:
        CsLie = sorted({str(C).zfill(6) for C in all_case_ids if str(C).isdigit()}, key=lambda x: int(x))
        for CsJ in CsLie:
            if CsJ in AlJg:
                HangLi.append(AlJg[CsJ])
            else:
                HangLi.append(KongHang(str(int(CsJ))))
    else:
        for CsJ in sorted(AlJg.keys(), key=lambda x: int(x)):
            HangLi.append(AlJg[CsJ])

    ZongJg = PgYiAl(SuoYZh)
    ZongJg["case_id"] = "ALL"
    print(f"[DONE] {len(HangLi)} cases, valid_all={ZongJg['valid_all']:.4f}")

    CunCsv(HangLi, ZongJg, ScLj)


def main():
    run(RuCsv, ScCsv)


if __name__ == "__main__":
    main()
