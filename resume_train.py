from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    model = YOLO(r"G:\VScode\fire_yolo_webcam\runs\detect\train\weights\last.pt")
    model.train(resume=True)

if __name__ == "__main__":
    freeze_support()
    main()