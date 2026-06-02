# YOLO 2D detector, output 2d_results.csv
import argparse
import csv
import os
import glob
import torch
from ultralytics import YOLO

# set paths
GenMu = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
YoloMu = os.path.join(GenMu, "data", "yolo_data")
JianCeMu = os.path.join(GenMu, "output", "detections")
MoXingMr = os.path.join(GenMu, "src", "yolo", "yolo11s-pose.pt")
MoXingCu = os.path.join(GenMu, "models")
ZhiXinYz = 0.25
IouYuZhi = 0.45
TuDaXiao = 640
ShuChuCsv = os.path.join(JianCeMu, "2d_results.csv")

# set csv columns
LieMing = [
    "filename",
    "L1_x", "L1_y", "L2_x", "L2_y", "R1_x", "R1_y", "R2_x", "R2_y",
    "L_bbox_TL_x", "L_bbox_TL_y", "L_bbox_TR_x", "L_bbox_TR_y",
    "L_bbox_BR_x", "L_bbox_BR_y", "L_bbox_BL_x", "L_bbox_BL_y",
    "R_bbox_TL_x", "R_bbox_TL_y", "R_bbox_TR_x", "R_bbox_TR_y",
    "R_bbox_BR_x", "R_bbox_BR_y", "R_bbox_BL_x", "R_bbox_BL_y",
    "L_conf", "L1_conf", "L2_conf", "R_conf", "R1_conf", "R2_conf",
]


def XuanMoXi(MxLuJi=None):
    # choose model path
    if MxLuJi and os.path.exists(MxLuJi):
        return MxLuJi
    if os.path.exists(MoXingMr):
        return MoXingMr
    return "yolo11s-pose.pt"

# save results to csv
def CunCsv(HangLie, LuJing):
    os.makedirs(os.path.dirname(LuJing), exist_ok=True)
    with open(LuJing, "w", newline="", encoding="utf-8") as fp:
        XieRu = csv.DictWriter(fp, fieldnames=LieMing)
        XieRu.writeheader()
        XieRu.writerows(HangLie)
    print(f"[DONE] saved: {LuJing}")

# collect image files from directory
def ShouJiTu(TuMuLu):
    TuLie = []
    for HouZhui in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        TuLie.extend(glob.glob(os.path.join(TuMuLu, HouZhui)))
    TuLie.sort()
    return TuLie

# extract detected objects from yolo result
def TiQuMuBi(JieGuo):
    WuTiLi = []
    if JieGuo.boxes is None or len(JieGuo.boxes) == 0:
        return WuTiLi
    KuangSj = JieGuo.boxes.xyxy.cpu().numpy()
    ZhiXinDu = JieGuo.boxes.conf.cpu().numpy()
    GjDianSj = None
    if hasattr(JieGuo, "keypoints") and JieGuo.keypoints is not None:
        GjDianSj = JieGuo.keypoints.data.cpu().numpy()
    for SuoYin in range(len(KuangSj)):
        WuTiGe = {"bbox": KuangSj[SuoYin], "conf": float(ZhiXinDu[SuoYin]), "kpts": []}
        if GjDianSj is not None and SuoYin < len(GjDianSj):
            DianShu = min(2, len(GjDianSj[SuoYin]))
            for Jg in range(DianShu):
                KpGe = GjDianSj[SuoYin][Jg]
                WuTiGe["kpts"].append((float(KpGe[0]), float(KpGe[1]),
                                       float(KpGe[2]) if len(KpGe) > 2 else 1.0))
        WuTiLi.append(WuTiGe)
    return WuTiLi

# assign left/right tools by x position
def FenPeiZy(WuTiLi):
    if len(WuTiLi) == 0:
        return None, None
    if len(WuTiLi) == 1:
        return WuTiLi[0], None
    PaiXu = sorted(WuTiLi, key=lambda Og: Og["conf"], reverse=True)[:2]
    PaiXu = sorted(PaiXu, key=lambda Og: (Og["bbox"][0] + Og["bbox"][2]) / 2.0)
    return PaiXu[0], PaiXu[1]

# fill one side data into the row dictnory
def TianChHg(HangGe, WuTiGe, QianZh):
    if WuTiGe is None:
        return
    X1, Y1, X2, Y2 = WuTiGe["bbox"]
    HangGe[f"{QianZh}_bbox_TL_x"] = round(float(X1), 1)
    HangGe[f"{QianZh}_bbox_TL_y"] = round(float(Y1), 1)
    HangGe[f"{QianZh}_bbox_TR_x"] = round(float(X2), 1)
    HangGe[f"{QianZh}_bbox_TR_y"] = round(float(Y1), 1)
    HangGe[f"{QianZh}_bbox_BR_x"] = round(float(X2), 1)
    HangGe[f"{QianZh}_bbox_BR_y"] = round(float(Y2), 1)
    HangGe[f"{QianZh}_bbox_BL_x"] = round(float(X1), 1)
    HangGe[f"{QianZh}_bbox_BL_y"] = round(float(Y2), 1)
    HangGe[f"{QianZh}_conf"] = round(float(WuTiGe["conf"]), 4)
    DianLie = WuTiGe["kpts"]
    if len(DianLie) > 0:
        HangGe[f"{QianZh}1_x"] = round(float(DianLie[0][0]), 1)
        HangGe[f"{QianZh}1_y"] = round(float(DianLie[0][1]), 1)
        HangGe[f"{QianZh}1_conf"] = round(float(DianLie[0][2]), 4)
    if len(DianLie) > 1:
        HangGe[f"{QianZh}2_x"] = round(float(DianLie[1][0]), 1)
        HangGe[f"{QianZh}2_y"] = round(float(DianLie[1][1]), 1)
        HangGe[f"{QianZh}2_conf"] = round(float(DianLie[1][2]), 4)

# convert yolo result to csv row
def JgZhHang(JieGuo, WjMing):
    HangGe = {Kg: "" for Kg in LieMing}
    HangGe["filename"] = WjMing
    WuTiLi = TiQuMuBi(JieGuo)
    ZuoGe, YouGe = FenPeiZy(WuTiLi)
    TianChHg(HangGe, ZuoGe, "L")
    TianChHg(HangGe, YouGe, "R")
    return HangGe

# YOLO detector class
class YoloJcq:
    def __init__(self, MxLuJi=None, SheBei="auto"):
        if SheBei == "auto":
            self.SheBei = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.SheBei = SheBei
        JiaZai = XuanMoXi(MxLuJi)
        self.MoXing = YOLO(JiaZai)
        self.ZhiXin = ZhiXinYz
        self.IouYz = IouYuZhi
        self.TuCc = TuDaXiao
        print(f"[INFO] device: {self.SheBei}, model: {JiaZai}")

    def SzZhiXin(self, ZhiGe):
        self.ZhiXin = max(0.0, min(1.0, float(ZhiGe)))

    def SzIouYz(self, ZhiGe):
        self.IouYz = max(0.0, min(1.0, float(ZhiGe)))

    # train model
    def XunLian(self, SjYaml, LunShu=80, PiCiSh=16, MingChen="tool_pose"):
        print("\n", "-" * 5, "Training YOLO", "-" * 5)
        if not os.path.exists(SjYaml):
            print(f"[ERROR] yaml not found: {SjYaml}")
            return None
        os.makedirs(MoXingCu, exist_ok=True)
        self.MoXing.train(
            data=SjYaml, epochs=int(LunShu), batch=int(PiCiSh),
            imgsz=self.TuCc, conf=self.ZhiXin, iou=self.IouYz,
            project=MoXingCu, name=MingChen, device=self.SheBei, exist_ok=True,
        )
        ZuiJiaLj = os.path.join(MoXingCu, MingChen, "weights", "best.pt")
        print(f"[DONE] best.pt: {ZuiJiaLj}")
        return ZuiJiaLj

    # batch inference on image directory
    def PiLiJc(self, TuMuLu):
        print("\n", "-" * 5, "Running YOLO 2D", "-" * 5)
        if not os.path.isdir(TuMuLu):
            print(f"[ERROR] dir not found: {TuMuLu}")
            return []
        TuLie = ShouJiTu(TuMuLu)
        if not TuLie:
            print(f"[ERROR] no images in {TuMuLu}")
            return []
        HangLie = []
        for SuoYin, TuLuJi in enumerate(TuLie):
            JieGuo = self.MoXing.predict(
                source=TuLuJi, conf=self.ZhiXin, iou=self.IouYz,
                imgsz=self.TuCc, device=self.SheBei, verbose=False,
            )[0]
            HangLie.append(JgZhHang(JieGuo, os.path.basename(TuLuJi)))
            if (SuoYin + 1) % 200 == 0:
                print(f"[INFO] progress: {SuoYin + 1}/{len(TuLie)}")
        print(f"[DONE] infer: {len(HangLie)} rows")
        return HangLie

# run test detection
def YunXingCs(MxLuJi=None, TuMuLu=None, ShuChuLj=None, ZhiXin=ZhiXinYz, IouZhi=IouYuZhi):
    if TuMuLu is None:
        TuMuLu = os.path.join(YoloMu, "images", "test")
    if ShuChuLj is None:
        ShuChuLj = ShuChuCsv
    JianCeQi = YoloJcq(MxLuJi=MxLuJi)
    JianCeQi.SzZhiXin(ZhiXin)
    JianCeQi.SzIouYz(IouZhi)
    HangLie = JianCeQi.PiLiJc(TuMuLu)
    if len(HangLie) == 0:
        return HangLie
    print("\n", "-" * 5, "Saving CSV", "-" * 5)
    CunCsv(HangLie, ShuChuLj)
    return HangLie

# backward compatible aliases
YOLO_CONF_THR = ZhiXinYz
YOLO_IOU_THR = IouYuZhi
YoloDetector = YoloJcq


def run_test_detection(model_path=None, image_dir=None, out_csv=None, conf=YOLO_CONF_THR, iou=YOLO_IOU_THR):
    return YunXingCs(MxLuJi=model_path, TuMuLu=image_dir, ShuChuLj=out_csv, ZhiXin=conf, IouZhi=iou)


def main():

    JieCi = argparse.ArgumentParser()
    JieCi.add_argument("--mode", type=str, default="detect", choices=["detect", "train"])
    JieCi.add_argument("--model", type=str, default=None)
    JieCi.add_argument("--input", type=str, default=None)
    JieCi.add_argument("--output", type=str, default=ShuChuCsv)
    JieCi.add_argument("--conf", type=float, default=ZhiXinYz)
    JieCi.add_argument("--iou", type=float, default=IouYuZhi)
    JieCi.add_argument("--data_yaml", type=str, default=os.path.join(YoloMu, "dataset.yaml"))
    JieCi.add_argument("--epochs", type=int, default=80)
    JieCi.add_argument("--batch", type=int, default=16)
    JieCi.add_argument("--name", type=str, default="tool_pose")
    CanShu = JieCi.parse_args()
    if CanShu.mode == "train":
        JianCeQi = YoloJcq(MxLuJi=CanShu.model)
        JianCeQi.SzZhiXin(CanShu.conf)
        JianCeQi.SzIouYz(CanShu.iou)
        JianCeQi.XunLian(CanShu.data_yaml, LunShu=CanShu.epochs, PiCiSh=CanShu.batch, MingChen=CanShu.name)
        return
    YunXingCs(MxLuJi=CanShu.model, TuMuLu=CanShu.input, ShuChuLj=CanShu.output,
              ZhiXin=CanShu.conf, IouZhi=CanShu.iou)


if __name__ == "__main__":
    main()
