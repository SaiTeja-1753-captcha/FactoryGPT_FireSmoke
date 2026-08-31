from ultralytics import YOLO
import cv2
from pathlib import Path

# ============================================================
# LOAD MODELS
# ============================================================

ppe_model = YOLO(r"model\best.pt")
firesmoke_model = YOLO(r"model\FireSmoke_best.pt")

print("PPE model loaded")
print("PPE classes:", ppe_model.names)

print("\nFire/Smoke model loaded")
print("Fire/Smoke classes:", firesmoke_model.names)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("\nWebcam started.")
print("Press Q to quit.")


# ============================================================
# OUTPUT FOLDER
# ============================================================

output_folder = Path("IntegratedResults")
output_folder.mkdir(exist_ok=True)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break


    # ========================================================
    # 1. PERSON TRACKING USING BYTETRACK
    # ========================================================

    tracking_results = ppe_model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        classes=[11],
        verbose=False
    )


    # ========================================================
    # 2. DRAW PERSON TRACKING
    # ========================================================

    if tracking_results and tracking_results[0].boxes is not None:

        boxes = tracking_results[0].boxes

        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = ppe_model.names[class_id]

            # ByteTrack ID
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1

            label = f"Person ID:{track_id} {confidence:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


    # ========================================================
    # 3. PPE DETECTION
    # ========================================================

    ppe_results = ppe_model.predict(
        source=frame,
        conf=0.25,
        verbose=False
    )


    print("\n========== PPE ==========")

    for result in ppe_results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = ppe_model.names[class_id]

            print(
                f"{class_name} | "
                f"Confidence: {confidence:.2f}"
            )

            # Don't draw Person again because
            # ByteTrack already draws the tracked person.
            if class_name == "Person":
                continue

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )

            label = f"{class_name} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )


    # ========================================================
    # 4. FIRE / SMOKE DETECTION
    # ========================================================

    fire_results = firesmoke_model.predict(
        source=frame,
        conf=0.25,
        verbose=False
    )


    print("\n========== FIRE / SMOKE ==========")

    for result in fire_results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = firesmoke_model.names[class_id]

            print(
                f"{class_name} | "
                f"Confidence: {confidence:.2f}"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            label = f"{class_name} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )


    # ========================================================
    # 5. DISPLAY
    # ========================================================

    cv2.imshow(
        "IntegrationTest - YOLO + ByteTrack + Fire/Smoke",
        frame
    )


    # ========================================================
    # 6. QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\nTracking stopped.")