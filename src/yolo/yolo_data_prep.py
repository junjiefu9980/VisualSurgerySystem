# prepare YOLO dataset which from frame_table.csv
import os
import cv2
import yaml
import argparse
import csv
import json
import random
import shutil

# set project root
GenMu = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ShuChu = os.path.join(GenMu, "output")
YoloMu = os.path.join(GenMu, "data", "yolo_data")
ZhenCsv = os.path.join(ShuChu, "frames", "frame_table.csv")

# set allowed training cases
HeGeAl = [
    "000000", "000001", "000002", "000003", "000004",
    "000006", "000007", "000008", "000009", "000010",
    "000011", "000012", "000013", "000014", "000015", "000016",
]

# set keypoint ids per tool
GjYiId = ["3", "4"]
GjErId = ["10", "11"]
MoRenKu = 1280
MoRenGa = 1024

# load csv rows
def DuCsv(LuJing):
    JiLu = []
    if not os.path.exists(LuJing):
        print(f" [ERROR] csv not found: {LuJing}")
        return JiLu
    with open(LuJing, "r", encoding="utf-8") as fp:
        for HangGe in csv.DictReader(fp):
            JiLu.append(HangGe)
    print(f"[INFO] loaded {len(JiLu)} rows")
    return JiLu

# parse json string
def JieXiJs(WenBen):
    try:
        return json.loads(WenBen)
    except Exception:
        return {}

# check bbox valid
def KuangOk(KuangGe):
    if not isinstance(KuangGe, (list, tuple)) or len(KuangGe) < 4:
        return False
    return KuangGe[2] > 0 and KuangGe[3] > 0

# clip value to [0,1]
def CaiJian(ZhiGe):
    if ZhiGe < 0.0:
        return 0.0
    if ZhiGe > 1.0:
        return 1.0
    return ZhiGe

# convert bbox to yolo format style
def BbZhYo(KuangGe, KuanDu, GaoDu):
    Xg, Yg, Wg, Hg = KuangGe[:4]
    Cx = CaiJian((Xg + Wg / 2) / KuanDu)
    Cy = CaiJian((Yg + Hg / 2) / GaoDu)
    Nw = CaiJian(Wg / KuanDu)
    Nh = CaiJian(Hg / GaoDu)
    return Cx, Cy, Nw, Nh

# convert keypoints to yolo format style
def KpZhYo(DianJi, IdLie, KuanDu, GaoDu):
    JieGuo = []
    for KgId in IdLie:
        if KgId in DianJi and DianJi[KgId] is not None:
            Px, Py = DianJi[KgId]
            JieGuo.append((CaiJian(Px / KuanDu), CaiJian(Py / GaoDu), 2))
        else:
            JieGuo.append((0.0, 0.0, 0))
    return JieGuo

# build yolo label line
def PinJieH(LeiBie, KuangGe, DianLie):
    Cx, Cy, Wg, Hg = KuangGe
    BuFen = [str(LeiBie), f"{Cx:.6f}", f"{Cy:.6f}", f"{Wg:.6f}", f"{Hg:.6f}"]
    for Nx, Ny, Vg in DianLie:
        BuFen.append(f"{Nx:.6f}")
        BuFen.append(f"{Ny:.6f}")
        BuFen.append(str(Vg))
    return " ".join(BuFen)

# process one frame, build label lines
def ChuLiZh(HangGe, TuKuan, TuGao):
    BbSj = JieXiJs(HangGe.get("boundingbox_raw", "{}"))
    KpSj = JieXiJs(HangGe.get("keypoints_raw", "{}"))
    JieGuo = []
    if "obj1" in BbSj and KuangOk(BbSj["obj1"]):
        YoKu = BbZhYo(BbSj["obj1"], TuKuan, TuGao)
        YoDi = KpZhYo(KpSj, GjYiId, TuKuan, TuGao)
        JieGuo.append(PinJieH(0, YoKu, YoDi))
    if "obj2" in BbSj and KuangOk(BbSj["obj2"]):
        YoKu = BbZhYo(BbSj["obj2"], TuKuan, TuGao)
        YoDi = KpZhYo(KpSj, GjErId, TuKuan, TuGao)
        JieGuo.append(PinJieH(0, YoKu, YoDi))
    return JieGuo

# split data by case
def FenGeSj(JiLu, BiLi=0.8, ZhZi=42):
    AnLiZu = {}
    for RgHang in JiLu:
        AlId = RgHang.get("case_id", "")
        if AlId not in AnLiZu:
            AnLiZu[AlId] = []
        AnLiZu[AlId].append(RgHang)
    AlLie = sorted(AnLiZu.keys())
    random.seed(ZhZi)
    random.shuffle(AlLie)
    NxShu = int(len(AlLie) * BiLi)
    XunJi = set(AlLie[:NxShu])
    XunLie = []
    YanLie = []
    for AlId, JlLie in AnLiZu.items():
        if AlId in XunJi:
            XunLie.extend(JlLie)
        else:
            YanLie.extend(JlLie)
    return XunLie, YanLie

# clean all files in directory
def QkMuLu(MuLuLj):
    if not os.path.isdir(MuLuLj):
        return
    for WjGe in os.listdir(MuLuLj):
        WjLuJi = os.path.join(MuLuLj, WjGe)
        if os.path.isfile(WjLuJi):
            os.remove(WjLuJi)

# export one split to images and labels
def DaoChFz(JiLu, FzMing, ShuChuMu):
    TuMuLu = os.path.join(ShuChuMu, "images", FzMing)
    BzMuLu = os.path.join(ShuChuMu, "labels", FzMing)
    os.makedirs(TuMuLu, exist_ok=True)
    os.makedirs(BzMuLu, exist_ok=True)
    OkShu = 0
    QueShi = 0

    for HangGe in JiLu:
        TuLuJi = HangGe.get("frame_path", "").replace("\\", "/")
        YuanTu = os.path.join(GenMu, TuLuJi)
        if not os.path.exists(YuanTu):
            YuanTu = TuLuJi
        if not os.path.exists(YuanTu):
            QueShi += 1
            continue
        WjMing = f"{HangGe['case_id']}_{HangGe['name']}_{HangGe['frame_idx']}"
        shutil.copy(YuanTu, os.path.join(TuMuLu, f"{WjMing}.jpg"))

        KuanDu, GaoDu = MoRenKu, MoRenGa
        TuPian = cv2.imread(YuanTu)
        if TuPian is not None:
            GaoDu, KuanDu = TuPian.shape[:2]
        BzHang = ChuLiZh(HangGe, KuanDu, GaoDu)
        with open(os.path.join(BzMuLu, f"{WjMing}.txt"), "w") as fp:
            fp.write("\n".join(BzHang))
        OkShu += 1
    print(f"[INFO] {FzMing}: done={OkShu} miss={QueShi}")

# write dataset(.yaml)
def XieYaml(ShuChuMu):
    NeiRong = {
        "path": os.path.abspath(ShuChuMu),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {0: "tool"}, "kpt_shape": [2, 3], "flip_idx": [0, 1],
    }
    with open(os.path.join(ShuChuMu, "dataset.yaml"), "w") as fp:
        yaml.dump(NeiRong, fp, sort_keys=False)

# build train/val/test dataset
def CjXySjj(CsvLuJi, ShuChuMu):
    print("\n", "-" * 5, "Build train/val/test", "-" * 5)
    JiLu = DuCsv(CsvLuJi)
    if not JiLu:
        return
    JiLu = [Rg for Rg in JiLu if Rg.get("case_id") in HeGeAl]
    print(f"[INFO] after filter: {len(JiLu)} rows")
    XunLie, YanLie = FenGeSj(JiLu)
    for FzGe in ["train", "val", "test"]:
        QkMuLu(os.path.join(ShuChuMu, "images", FzGe))
        QkMuLu(os.path.join(ShuChuMu, "labels", FzGe))
    DaoChFz(XunLie, "train", ShuChuMu)
    DaoChFz(YanLie, "val", ShuChuMu)
    DaoChFz(YanLie, "test", ShuChuMu)
    XieYaml(ShuChuMu)
    print(f"[DONE] dataset: {ShuChuMu}")

# build unseen test set
def CjDlCs(CsvLuJi, ShuChuMu):
    print("\n", "-" * 5, "Build unseen test", "-" * 5)
    JiLu = DuCsv(CsvLuJi)
    if not JiLu:
        return
    SuoYAl = sorted({Rg.get("case_id", "") for Rg in JiLu})
    CeShiAl = [Ag for Ag in SuoYAl if Ag not in HeGeAl and Ag != "000005"]
    print(f"[INFO] unseen cases: {CeShiAl}")
    CeShiJl = [Rg for Rg in JiLu if Rg.get("case_id") in CeShiAl]
    QkMuLu(os.path.join(ShuChuMu, "images", "test"))
    QkMuLu(os.path.join(ShuChuMu, "labels", "test"))
    DaoChFz(CeShiJl, "test", ShuChuMu)
    print("[DONE] unseen test updated")


def main():
    JieCi = argparse.ArgumentParser()
    JieCi.add_argument("--csv", type=str, default=ZhenCsv)
    JieCi.add_argument("--out", type=str, default=YoloMu)
    JieCi.add_argument("--mode", type=str, default="trainval", choices=["trainval", "test", "both"])
    CanShu = JieCi.parse_args()
    if CanShu.mode in ("trainval", "both"):
        CjXySjj(CanShu.csv, CanShu.out)
    if CanShu.mode in ("test", "both"):
        CjDlCs(CanShu.csv, CanShu.out)


if __name__ == "__main__":
    main()
