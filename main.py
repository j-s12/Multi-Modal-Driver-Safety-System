import cv2
from ultralytics import YOLO
import numpy as np
import requests
import winsound

#Telegram Alert Function
def send_alert(message):
    TOKEN =  "8779657290:AAFJhEo2mI6NL-u4Cro4QXtgWTs_zVfI4vQ"
    CHAT_ID =  "1794753837"

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}

    try:
        requests.post(url, data=data)
    except:
        pass


#Load model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("road.mp4")
cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

closed_frames = 0 

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

#Save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (frame_width, frame_height))

cv2.namedWindow("ADAS FINAL CLEAN", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ADAS FINAL CLEAN", 800, 500)

#DRIVER WINDOW
cv2.namedWindow("Driver Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Driver Monitor", 800, 500)

#Tracking
object_id = 0
centers = {}

#Performance
frame_count = 0
last_results = None
last_alert_frame = 0


#Lane Detection
def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(frame)

    if brightness < 50:
        gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 30, 100)
    else:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape

    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width, int(height * 0.6)),
        (0, int(height * 0.6))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped = cv2.bitwise_and(edges, mask)

    if brightness < 50:
        lines = cv2.HoughLinesP(cropped, 1, np.pi/180, 120,
                                minLineLength=100, maxLineGap=20)
    else:
        lines = cv2.HoughLinesP(cropped, 1, np.pi/180, 50,
                                minLineLength=50, maxLineGap=50)

    return lines


while cap.isOpened():
    ret, frame = cap.read()
    ret2, driver_frame = cam.read()  # 👈 ADDED

    if not ret or not ret2:
        break

    frame_count += 1

    if frame_count % 2 != 0:
        continue

    annotated_frame = frame.copy()

    brightness = np.mean(frame)
    night_mode = brightness < 50

    risk_score = 0

    #YOLO optimization
    if frame_count % 3 == 0:
        results = model(frame)
        last_results = results
    else:
        results = last_results

    if results is None:
        continue

    #OBJECT DETECTION
    for box in results[0].boxes:
        cls = int(box.cls[0])

        if cls not in [2, 5, 7]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        width = x2 - x1
        height = y2 - y1
        area = width * height

        distance = int(10000 / (width + 1))

        if distance < 50:
            risk_score += 30

        if area > 20000:
            risk_score += 40

        #TRACKING
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        found = False
        for id, (px, py) in centers.items():
            if abs(cx - px) < 50 and abs(cy - py) < 50:
                centers[id] = (cx, cy)
                track_id = id
                found = True
                break

        if not found:
            object_id += 1
            centers[object_id] = (cx, cy)
            track_id = object_id

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(annotated_frame,
                    f"ID:{track_id} {distance}m",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2)

        if area > 20000:
            cv2.putText(annotated_frame,
                        "WARNING: TOO CLOSE!",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2)

    #LANES
    lines = detect_lanes(frame)

    lane_center = frame.shape[1] // 2
    left_x = []
    right_x = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if x1 < lane_center and x2 < lane_center:
                left_x.extend([x1, x2])
            elif x1 > lane_center and x2 > lane_center:
                right_x.extend([x1, x2])

        if left_x and right_x:
            left_avg = int(sum(left_x) / len(left_x))
            right_avg = int(sum(right_x) / len(right_x))

            lane_mid = (left_avg + right_avg) // 2

            cv2.line(annotated_frame,
                     (lane_mid, 0),
                     (lane_mid, frame.shape[0]),
                     (255, 0, 0), 2)

            if abs(lane_mid - lane_center) > 50:
                risk_score += 30
                cv2.putText(annotated_frame,
                            "WARNING: Lane Departure!",
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2)

    #FATIGUE DETECTION
    gray = cv2.cvtColor(driver_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_detected = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

        if len(eyes) > 0:
            eyes_detected = True

    if not eyes_detected:
        closed_frames += 1
    else:
        closed_frames = 0

    drowsy = closed_frames > 20

    if drowsy:
        cv2.putText(driver_frame, "DROWSY ALERT!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #ALERT
    if risk_score >= 60 and frame_count - last_alert_frame > 50:
        winsound.Beep(1000, 300)
        send_alert("⚠️ HIGH RISK DETECTED! Check surroundings.")
        last_alert_frame = frame_count

    if drowsy and frame_count - last_alert_frame > 50:
        winsound.Beep(1000, 300)
        send_alert("😴 DRIVER DROWSINESS DETECTED!")
        last_alert_frame = frame_count

    #UI
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_width, 80), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

    cv2.putText(annotated_frame,
                "ADAS SYSTEM",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2)

    mode_text = "NIGHT MODE" if night_mode else "DAY MODE"
    cv2.putText(annotated_frame,
                f"Mode: {mode_text}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2)

    display_frame = cv2.resize(annotated_frame, (800, 500))
    cv2.imshow("ADAS FINAL CLEAN", display_frame)
    cv2.imshow("Driver Monitor", driver_frame) 

    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cam.release()
out.release()
cv2.destroyAllWindows()