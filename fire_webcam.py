import cv2
from config import *
from ultralytics import YOLO

# Đổi đường dẫn này thành model nhận diện lửa của bạn
# Ví dụ: best.pt sau khi train xong
model = YOLO(MODEL_PATH)

# Mở webcam laptop
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không mở được webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Không đọc được hình ảnh từ webcam")
        break

    # Nhận diện bằng YOLO
    results = model.predict(
        source=frame,
        conf=0.5,
        imgsz=640,
        verbose=False
    )

    # Vẽ khung nhận diện
    annotated_frame = results[0].plot()

    # Hiển thị
    cv2.imshow("Fire Detection YOLO", annotated_frame)

    # Nhấn q để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()