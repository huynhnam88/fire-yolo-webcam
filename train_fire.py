from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data=r"G:\VScode\fire_yolo_webcam\fire-detection-yolo-2\data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        save=True,
        save_period=1
    )

if __name__ == "__main__":
    freeze_support()
    main()