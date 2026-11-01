"""Train a YOLO11 detector with Ultralytics, resuming from the last checkpoint if there is one.

    python train_yolo.py --data VOC.yaml --model yolo11n.pt --epochs 100 --run-dir /scratch/$USER/runs/yolo-voc
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO, settings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="VOC.yaml", help="VOC.yaml, coco128.yaml, or your own dataset YAML")
    p.add_argument("--model", default="yolo11n.pt", help="yolo11n/s/m/l/x.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--run-dir", type=Path, required=True, help="checkpoints and results")
    p.add_argument("--datasets", type=Path, default=Path(f"/scratch/{os.environ['USER']}/datasets"))
    a = p.parse_args()

    # Ultralytics downloads the built-in datasets (VOC, COCO, ...) here, once per node.
    settings.update({"datasets_dir": str(a.datasets)})

    last = a.run_dir / "train" / "weights" / "last.pt"
    if last.exists():
        print(f"Resuming from {last}")
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(a.model).train(
            data=a.data,
            epochs=a.epochs,
            imgsz=a.imgsz,
            batch=a.batch,
            workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)),
            project=str(a.run_dir),
            name="train",
            exist_ok=True,
            device=0,
        )

    best = a.run_dir / "train" / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=a.data, imgsz=a.imgsz, device=0)
    print(f"best.pt  mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
