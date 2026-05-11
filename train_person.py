# train_person.py
import os
from ultralytics import YOLO

DATA_YAML = "/content/drive/MyDrive/person/data.yaml"      # <- path to your YAML
BASE_MODEL = "yolov8n.pt"      # n=speed, s/m=little heavier; swap to 'yolov8s.pt' if you have a good GPU
IMGSZ = 640
EPOCHS = 120
BATCH = 16                     # tune per your GPU RAM (try 32 if you can)
PROJECT = "/content/drive/MyDrive/personruns"
RUN_NAME = "person_yolov8n"

if __name__ == "__main__":
    # 1) Load a pretrained COCO model (has 'person' prior) and fine-tune
    model = YOLO(BASE_MODEL)

    # 2) Train
    results = model.train(
        data=DATA_YAML,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,# 0 for first GPU; set "cpu" to train on CPU (slow)
        workers=8,
        optimizer="Adamw",      # good default
        lr0=0.001, lrf=0.01,    # cosine final LR multiplier
        momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        # Data aug (safe defaults; reduce if overfitting)
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5, translate=0.10, scale=0.50, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.10, copy_paste=0.0,
        close_mosaic=10,        # disable mosaic in last epochs for quality
        patience=30,            # early stop if no val improvement
        project=PROJECT, name=RUN_NAME, exist_ok=True
    )

    # 3) Validate once more (saves PR/ROC curves etc.)
    model.val(data=DATA_YAML, imgsz=IMGSZ,)

    # 4) Export formats you might use later
    #   - .pt for Raspberry Pi (PyTorch)
    #   - .onnx optional
    model.export(format="pt")                          # best.pt in runs/detect/<RUN_NAME>/weights
    model.export(format="onnx", opset=12, simplify=True)
    # (Optional) Other targets if you need them:
    # model.export(format="ncnn")
    # model.export(format="tflite")  # may require extra deps
