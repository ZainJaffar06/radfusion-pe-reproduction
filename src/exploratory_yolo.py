"""EXPLORATORY imaging arm — NOT part of the reported fusion.

This trains a lightweight YOLOv8n-cls classifier on CT slices as a candidate compact imaging backbone
for future work. It is evaluated at the SLICE level only. Its scores are **not** aggregated to study
level and **not** combined into any fusion result reported in the manuscript; every reported imaging
and fusion number uses the RadFusion-provided vision-model score instead (see src/data.py and README).

Under strong class imbalance, slice-level accuracy approaches the majority-negative base rate and is
not a meaningful measure of PE detection; treat any slice-level number here as exploratory only.

Expected data layout (Ultralytics classification format), which you must prepare from the CTPA slices:
    slices/
      train/{pe,no_pe}/*.png
      val/{pe,no_pe}/*.png
      test/{pe,no_pe}/*.png
"""
from __future__ import annotations
import argparse
from pathlib import Path


def train_exploratory(data_dir: str, epochs: int = 2, imgsz: int = 224, model: str = "yolov8n-cls.pt"):
    """Fine-tune YOLOv8n-cls on CT slices (exploratory; slice-level only)."""
    from ultralytics import YOLO  # local import; optional dependency

    net = YOLO(model)
    results = net.train(data=data_dir, epochs=epochs, imgsz=imgsz, seed=42, verbose=True)
    print(
        "\nNOTE: this is an exploratory SLICE-level classifier. Its predictions were NOT aggregated to "
        "study level and NOT used in any reported fusion result. Do not report these numbers as the "
        "imaging or fusion performance of the paper."
    )
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Exploratory YOLOv8n-cls slice classifier (not fused)")
    ap.add_argument("--data", required=True, help="path to the slice dataset (Ultralytics cls layout)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=224)
    args = ap.parse_args()
    train_exploratory(args.data, epochs=args.epochs, imgsz=args.imgsz)
