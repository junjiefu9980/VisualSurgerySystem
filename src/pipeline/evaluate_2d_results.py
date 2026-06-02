# evaluate 2D detection by precision/recall/mAP.

import csv
import json
import math
import re
import os


# paths
GenMl = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ScMl = os.path.join(GenMl, "output")
JcMl = os.path.join(ScMl, "detections")
PgMl = os.path.join(ScMl, "evaluation")

GtCsv = os.path.join(ScMl, "frames", "frame_table.csv")
YcCsv = os.path.join(JcMl, "2d_results.csv")
ScCsv = os.path.join(PgMl, "evaluate_2d_results.csv")

# confidence threshold
ZxYz = 0.25
# iou threshold
IouYz = 0.50


# safe float
def AnQFlt(X, MoRen=None):
    try:
        if X is None:
            return MoRen
        if isinstance(X, float) and math.isnan(X):
            return MoRen
        return float(X)
    except (TypeError, ValueError):
        return MoRen

# normalize bbox to xyxy
def GuiFanBb(X1, Y1, X2, Y2):
    return [float(min(X1, X2)), float(min(Y1, Y2)),
            float(max(X1, X2)), float(max(Y1, Y2))]

# bbox area
def BbMianJi(B):
    W = max(0.0, B[2] - B[0])
    H = max(0.0, B[3] - B[1])
    return W * H

# IoU of two bboxes
def SuanIou(A, B):
    Ix1 = max(A[0], B[0])
    Iy1 = max(A[1], B[1])
    Ix2 = min(A[2], B[2])
    Iy2 = min(A[3], B[3])

    Iw = max(0.0, Ix2 - Ix1)
    Ih = max(0.0, Iy2 - Iy1)
    JiaoJi = Iw * Ih

    MjA = BbMianJi(A)
    MjB = BbMianJi(B)
    BingJi = MjA + MjB - JiaoJi
    if BingJi <= 0.0:
        return 0.0
    return JiaoJi / BingJi

# make image id
def ShCTuId(Cid, MingZi, ZhHao):
    return f"{int(Cid):06d}_{MingZi}_{int(ZhHao)}"

# parse prediction filename
def JxWjMing(WjMing):
    Jg = os.path.splitext(WjMing)[0]
    BuFen = Jg.split("_")
    if len(BuFen) < 3:
        return None
    if not re.fullmatch(r"\d{6}", BuFen[0]):
        return None
    Cid = int(BuFen[0])
    ZhHao = int(BuFen[-1])
    MingZi = "_".join(BuFen[1:-1])
    return Cid, MingZi, ZhHao

# extract corners to xyxy
def JqZhXy(Hang, QianZh):
    XLie = []
    YLie = []
    for Jiao in ["TL", "TR", "BR", "BL"]:
        Xv = AnQFlt(Hang.get(f"{QianZh}_bbox_{Jiao}_x"))
        Yv = AnQFlt(Hang.get(f"{QianZh}_bbox_{Jiao}_y"))
        if Xv is None or Yv is None:
            return None
        XLie.append(Xv)
        YLie.append(Yv)
    return [min(XLie), min(YLie), max(XLie), max(YLie)]

# load GT data by case
def DuGtSj(GtLj):
    print("\n", "-" * 5, "Step 1: Loading GT", "-" * 5)
    if not os.path.exists(GtLj):
        print(f"[Warning] {GtLj} not exist")
        return {}

    GtMap = {}
    ZongHg = 0
    YxHg = 0
    ZongKg = 0

    with open(GtLj, "r") as f:
        for Hang in csv.DictReader(f):
            ZongHg += 1
            Cid = int(Hang["case_id"])
            MingZi = Hang["name"]
            ZhHao = int(Hang["frame_idx"])
            TuId = ShCTuId(Cid, MingZi, ZhHao)

            try:
                Bb = json.loads(Hang["boundingbox_raw"])
            except json.JSONDecodeError:
                Bb = {}

            KuangLi = []
            for K in ("obj1", "obj2"):
                V = Bb.get(K, None)
                if not isinstance(V, (list, tuple)) or len(V) < 4:
                    continue
                X, Y, W, H = V[:4]
                Xf = AnQFlt(X, 0.0) or 0.0
                Yf = AnQFlt(Y, 0.0) or 0.0
                Wf = AnQFlt(W, 0.0) or 0.0
                Hf = AnQFlt(H, 0.0) or 0.0
                B = GuiFanBb(Xf, Yf, Xf + Wf, Yf + Hf)
                if BbMianJi(B) > 1.0:
                    KuangLi.append(B)

            if Cid not in GtMap:
                GtMap[Cid] = {}
            GtMap[Cid][TuId] = KuangLi
            YxHg += 1
            ZongKg += len(KuangLi)

    NTu = sum(len(V) for V in GtMap.values())
    print(f"[DONE] GT: cases={len(GtMap)}, imgs={NTu}, boxes={ZongKg}")
    return GtMap

# load YOLO predictions
def DuYcSj(YcLj):
    print("\n", "-" * 5, "Step 2: Loading Predictions", "-" * 5)
    if not os.path.exists(YcLj):
        print(f"[WARNING] {YcLj} not exist")
        return {}

    YcMap = {}
    ZongHg = 0
    TiaoGo = 0

    with open(YcLj, "r", encoding="utf-8") as f:
        for Hang in csv.DictReader(f):
            ZongHg += 1
            Jx = JxWjMing(Hang.get("filename", ""))
            if Jx is None:
                TiaoGo += 1
                continue

            Cid, MingZi, ZhHao = Jx
            TuId = ShCTuId(Cid, MingZi, ZhHao)

            if Cid not in YcMap:
                YcMap[Cid] = {}
            if TuId not in YcMap[Cid]:
                YcMap[Cid][TuId] = []

            # extract L/R bbox
            for Qz in ["L", "R"]:
                Kuang = JqZhXy(Hang, Qz)
                ZxDu = AnQFlt(Hang.get(f"{Qz}_conf"))
                if Kuang is not None and ZxDu is not None and BbMianJi(Kuang) > 1.0:
                    YcMap[Cid][TuId].append({"bbox": Kuang, "conf": float(ZxDu)})

    NTu = sum(len(V) for V in YcMap.values())
    print(f"[DONE] Pred: cases={len(YcMap)}, imgs={NTu}, skip={TiaoGo}")
    return YcMap


# compute precision/recall/f1
def SuanPRF(GtTu, YcTu, ZxYz=0.25, IouYz=0.5):
    # collect all preds above thr
    SuoYYc = []
    for TuId, YcLie in YcTu.items():
        for P in YcLie:
            if P["conf"] >= ZxYz:
                SuoYYc.append((P["conf"], TuId, P["bbox"]))
    SuoYYc.sort(key=lambda x: -x[0])

    # match flags
    PiPei = {}
    for Tid in GtTu.keys():
        PiPei[Tid] = [False] * len(GtTu.get(Tid, []))

    Tp = 0
    Fp = 0

    for Zx, TuId, Kuang in SuoYYc:
        GtLie = GtTu.get(TuId, [])
        if len(GtLie) == 0:
            Fp += 1
            continue

        ZjIou = 0.0
        ZjIdx = -1
        for j, Gt in enumerate(GtLie):
            if PiPei[TuId][j]:
                continue
            V = SuanIou(Kuang, Gt)
            if V > ZjIou:
                ZjIou = V
                ZjIdx = j

        if ZjIou >= IouYz and ZjIdx >= 0:
            PiPei[TuId][ZjIdx] = True
            Tp += 1
        else:
            Fp += 1

    ZongGt = sum(len(V) for V in GtTu.values())
    Fn = ZongGt - Tp
    Prec = Tp / (Tp + Fp) if (Tp + Fp) > 0 else 0.0
    Rec = Tp / ZongGt if ZongGt > 0 else 0.0
    F1 = (2 * Prec * Rec / (Prec + Rec)) if (Prec + Rec) > 0 else 0.0
    return Tp, Fp, Fn, ZongGt, len(SuoYYc), Prec, Rec, F1


# compute AP from PR curve
def PrZhAp(RecLi, PreLi):
    if not RecLi:
        return 0.0
    MRec = [0.0] + list(RecLi) + [1.0]
    MPre = [0.0] + list(PreLi) + [0.0]

    i = len(MPre) - 2
    while i >= 0:
        if MPre[i] < MPre[i + 1]:
            MPre[i] = MPre[i + 1]
        i -= 1

    Ap = 0.0
    for i in range(len(MRec) - 1):
        if MRec[i + 1] != MRec[i]:
            Ap += (MRec[i + 1] - MRec[i]) * MPre[i + 1]
    return float(Ap)


# compute AP at given IoU
def SuanAp(GtTu, YcTu, IouYz=0.5):
    ZongGt = sum(len(V) for V in GtTu.values())
    if ZongGt <= 0:
        return 0.0

    SuoYYc = []
    for TuId, YcLie in YcTu.items():
        for P in YcLie:
            SuoYYc.append((P["conf"], TuId, P["bbox"]))
    SuoYYc.sort(key=lambda x: -x[0])

    PiPei = {}
    for Tid in GtTu.keys():
        PiPei[Tid] = [False] * len(GtTu.get(Tid, []))

    TpLie = []
    LjTp = 0
    LjFp = 0

    for Zx, TuId, Kuang in SuoYYc:
        GtLie = GtTu.get(TuId, [])
        if len(GtLie) == 0:
            LjFp += 1
            TpLie.append((LjTp, LjFp))
            continue

        ZjIou = 0.0
        ZjIdx = -1
        for j, Gt in enumerate(GtLie):
            if PiPei[TuId][j]:
                continue
            V = SuanIou(Kuang, Gt)
            if V > ZjIou:
                ZjIou = V
                ZjIdx = j

        if ZjIou >= IouYz and ZjIdx >= 0:
            PiPei[TuId][ZjIdx] = True
            LjTp += 1
        else:
            LjFp += 1
        TpLie.append((LjTp, LjFp))

    RecLi = []
    PreLi = []
    for Ti, Fi in TpLie:
        RecLi.append(Ti / ZongGt)
        PreLi.append(Ti / (Ti + Fi) if (Ti + Fi) > 0 else 0.0)
    return PrZhAp(RecLi, PreLi)


# evaluate by case
def PgAnLi(GtMap, YcMap, ZxYzV=0.25, IouYzV=0.5):
    print("\n", "-" * 5, "Step 3: Evaluating", "-" * 5)

    GjId = sorted(set(GtMap.keys()) & set(YcMap.keys()))
    if len(GjId) == 0:
        print("[Warning] No overlap cases")
        return [], None

    HangLi = []
    ZgGt = {}
    ZgYc = {}

    for Cid in GjId:
        GtTu = GtMap.get(Cid, {})
        YcTu = YcMap.get(Cid, {})
        GjTu = set(GtTu.keys()) & set(YcTu.keys())
        GtTu = {K: GtTu[K] for K in GjTu}
        YcTu = {K: YcTu[K] for K in GjTu}

        ZgGt.update(GtTu)
        ZgYc.update(YcTu)

        Tp, Fp, Fn, Ngt, Npd, P, R, F1 = SuanPRF(GtTu, YcTu, ZxYzV, IouYzV)
        Mp50 = SuanAp(GtTu, YcTu, IouYz=0.50)

        # mAP50-95
        ApLie = []
        T = 0.50
        while T < 0.96:
            ApLie.append(SuanAp(GtTu, YcTu, IouYz=round(T, 2)))
            T += 0.05
        Mp5095 = sum(ApLie) / len(ApLie) if ApLie else 0.0

        HangLi.append({
            "case_id": Cid, "images": len(GjTu),
            "gt_boxes": Ngt, "pred_boxes_confthr": Npd,
            "conf_thr": ZxYzV, "iou_thr": IouYzV,
            "tp": Tp, "fp": Fp, "fn": Fn,
            "precision": P, "recall": R, "f1": F1,
            "mAP50": Mp50, "mAP50-95": Mp5095,
        })

    # overall
    Tp, Fp, Fn, Ngt, Npd, P, R, F1 = SuanPRF(ZgGt, ZgYc, ZxYzV, IouYzV)
    Mp50 = SuanAp(ZgGt, ZgYc, IouYz=0.50)
    ApLie = []
    T = 0.50
    while T < 0.96:
        ApLie.append(SuanAp(ZgGt, ZgYc, IouYz=round(T, 2)))
        T += 0.05
    Mp5095 = sum(ApLie) / len(ApLie) if ApLie else 0.0

    ZongJg = {
        "case_id": "ALL", "images": len(ZgGt),
        "gt_boxes": Ngt, "pred_boxes_confthr": Npd,
        "conf_thr": ZxYzV, "iou_thr": IouYzV,
        "tp": Tp, "fp": Fp, "fn": Fn,
        "precision": P, "recall": R, "f1": F1,
        "mAP50": Mp50, "mAP50-95": Mp5095,
    }
    print(f"[DONE] cases={len(HangLi)}, P={P:.4f}, R={R:.4f}, mAP50={Mp50:.4f}")
    return HangLi, ZongJg


# save evaluation csv
def CunPgCsv(HangLi, ZongJg, ScLj):
    print("\n", "-" * 5, "Step 4: Saving", "-" * 5)
    os.makedirs(os.path.dirname(ScLj), exist_ok=True)

    LieMing = [
        "case_id", "images", "gt_boxes", "pred_boxes_confthr",
        "conf_thr", "iou_thr", "tp", "fp", "fn",
        "precision", "recall", "f1", "mAP50", "mAP50-95",
    ]

    with open(ScLj, "w", newline="", encoding="utf-8") as f:
        Xie = csv.DictWriter(f, fieldnames=LieMing)
        Xie.writeheader()
        for R in HangLi:
            Rr = dict(R)
            for K in ["precision", "recall", "f1", "mAP50", "mAP50-95"]:
                Rr[K] = round(Rr[K], 6)
            Xie.writerow(Rr)
        if ZongJg is not None:
            Oo = dict(ZongJg)
            for K in ["precision", "recall", "f1", "mAP50", "mAP50-95"]:
                Oo[K] = round(Oo[K], 6)
            Xie.writerow(Oo)
    print(f"[DONE] Saved: {ScLj}")


GT_CSV_PATH = GtCsv
PRED_CSV_PATH = YcCsv
OUT_CSV_PATH = ScCsv
CONF_THR = ZxYz
IOU_THR = IouYz
GT_CSV = GtCsv


def load_gt_by_case(GtLj):
    return DuGtSj(GtLj)


def load_pred_by_case(YcLj):
    return DuYcSj(YcLj)


def evaluate_by_case(GtMap, YcMap, conf_thr=CONF_THR, iou_thr=IOU_THR):
    return PgAnLi(GtMap, YcMap, ZxYzV=conf_thr, IouYzV=iou_thr)


def save_eval_csv(HangLi, ZongJg, ScLj):
    return CunPgCsv(HangLi, ZongJg, ScLj)


def main():
    GtMap = DuGtSj(GtCsv)
    YcMap = DuYcSj(YcCsv)
    HangLi, ZongJg = PgAnLi(GtMap, YcMap, ZxYzV=ZxYz, IouYzV=IouYz)
    CunPgCsv(HangLi, ZongJg, ScCsv)


# export aliases for pipeline
load_gt = DuGtSj
load_pred = DuYcSj
eval_cases = PgAnLi
save_eval = CunPgCsv

if __name__ == "__main__":
    main()
