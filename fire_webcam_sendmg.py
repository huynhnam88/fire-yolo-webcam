import cv2
import time
import requests
from config import *
from ultralytics import YOLO


# ================== CẤU HÌNH ==================


# Ngưỡng lọc báo cháy
CONF_THRESHOLD = 0.65       # độ tin cậy YOLO
FRAME_CONFIRM = 10          # phải phát hiện liên tục 10 frame mới báo
MIN_AREA_RATIO = 0.005      # vùng cháy tối thiểu = 0.5% khung hình
ALERT_COOLDOWN = 60         # 60 giây mới gửi lại 1 cảnh báo

# Class lửa trong model của bạn
FIRE_CLASSES = ["fire", "flame"]


# ================== HÀM GỬI TELEGRAM ==================

def send_telegram_alert(image_path, message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    try:
        with open(image_path, "rb") as photo:
            data = {
                "chat_id": CHAT_ID,
                "caption": message
            }

            files = {
                "photo": photo
            }

            response = requests.post(url, data=data, files=files, timeout=15)

        print("Telegram response:", response.text)

    except Exception as e:
        print("Lỗi gửi Telegram:", e)


# ================== CHƯƠNG TRÌNH CHÍNH ==================

def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_URL)

    if not cap.isOpened():
        print("Không kết nối được camera. Kiểm tra lại RTSP URL.")
        return

    fire_counter = 0
    last_alert_time = 0

    print("Đang chạy nhận diện lửa từ camera...")
    print("Nhấn Q để thoát.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Không đọc được frame từ camera. Đang thử lại...")
            time.sleep(1)
            continue

        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            imgsz=640,
            verbose=False
        )

        annotated_frame = results[0].plot()

        fire_detected_this_frame = False
        max_conf = 0
        max_area_ratio = 0

        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_w * frame_h

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id].lower()

                x1, y1, x2, y2 = box.xyxy[0]
                box_area = float((x2 - x1) * (y2 - y1))
                area_ratio = box_area / frame_area

                if conf > max_conf:
                    max_conf = conf
                    max_area_ratio = area_ratio

                if (
                    class_name in FIRE_CLASSES
                    and conf >= CONF_THRESHOLD
                    and area_ratio >= MIN_AREA_RATIO
                ):
                    fire_detected_this_frame = True
                    break

        if fire_detected_this_frame:
            fire_counter += 1
        else:
            fire_counter = max(0, fire_counter - 1)

        cv2.putText(
            annotated_frame,
            f"Fire counter: {fire_counter}/{FRAME_CONFIRM}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow("Fire Detection Camera", annotated_frame)

        current_time = time.time()

        if (
            fire_counter >= FRAME_CONFIRM
            and current_time - last_alert_time >= ALERT_COOLDOWN
        ):
            image_path = r"G:\VScode\fire_yolo_webcam\alerts\fire_alert.jpg"
            cv2.imwrite(image_path, annotated_frame)

            message = (
                "CẢNH BÁO CHÁY!\n"
                "Hệ thống YOLO phát hiện dấu hiệu ngọn lửa từ camera.\n"
                f"Độ tin cậy cao nhất: {max_conf:.2f}\n"
                f"Tỷ lệ vùng nghi cháy: {max_area_ratio * 100:.2f}%\n"
                "Vui lòng kiểm tra khu vực camera ngay."
            )

            send_telegram_alert(image_path, message)

            last_alert_time = current_time
            fire_counter = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()