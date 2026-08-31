# from ultralytics import YOLO
# import cv2

# # ============================================================
# # LOAD MODELS
# # ============================================================

# # PPE model
# ppe_model = YOLO(r"model\best.pt")

# # Fire + Smoke model
# fire_model = YOLO(r"model\fire_smoke\FireSmoke_best.pt")

# print("\n================ PPE MODEL ================")
# print(ppe_model.names)

# print("\n============= FIRE/SMOKE MODEL ============")
# print(fire_model.names)

# # ============================================================
# # CAMERA
# # ============================================================

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("ERROR: Camera could not be opened")
#     exit()

# # ============================================================
# # REAL-TIME DETECTION
# # ============================================================

# while True:

#     ret, frame = cap.read()

#     if not ret:
#         print("ERROR: Could not read camera frame")
#         break

#     # ========================================================
#     # PPE DETECTION
#     # ========================================================

#     ppe_results = ppe_model.predict(
#         source=frame,
#         conf=0.25,
#         imgsz=640,
#         verbose=False
#     )

#     ppe_result = ppe_results[0]

#     # ========================================================
#     # FIRE / SMOKE DETECTION
#     # ========================================================

#     fire_results = fire_model.predict(
#         source=frame,
#         conf=0.25,
#         imgsz=640,
#         verbose=False
#     )

#     fire_result = fire_results[0]

#     # ========================================================
#     # DRAW PPE DETECTIONS
#     # ========================================================

#     annotated = ppe_result.plot()

#     # ========================================================
#     # DRAW FIRE / SMOKE DETECTIONS
#     # ========================================================

#     if fire_result.boxes is not None:

#         for box in fire_result.boxes:

#             x1, y1, x2, y2 = map(
#                 int,
#                 box.xyxy[0].tolist()
#             )

#             class_id = int(box.cls[0])
#             confidence = float(box.conf[0])

#             class_name = fire_model.names[class_id]

#             label = f"{class_name} {confidence:.2f}"

#             # Fire/smoke bounding box
#             cv2.rectangle(
#                 annotated,
#                 (x1, y1),
#                 (x2, y2),
#                 (0, 0, 255),
#                 2
#             )

#             # Label
#             cv2.putText(
#                 annotated,
#                 label,
#                 (x1, max(y1 - 10, 20)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (0, 0, 255),
#                 2
#             )

#     # ========================================================
#     # DISPLAY
#     # ========================================================

#     cv2.imshow(
#         "FactoryGPT - PPE + Fire + Smoke Detection",
#         annotated
#     )

#     # ========================================================
#     # PRESS Q TO EXIT
#     # ========================================================

#     key = cv2.waitKey(1) & 0xFF

#     if key == ord("q"):
#         break

# # ============================================================
# # CLEANUP
# # ============================================================

# cap.release()
# cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2

# ============================================================
# MODEL PATHS
# ============================================================

PPE_MODEL_PATH = r"D:\FactoryGPT\model\best.pt"
FIRE_SMOKE_MODEL_PATH = r"D:\FactoryGPT\model\fire_smoke\FireSmoke_best.pt"

# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading PPE model...")
ppe_model = YOLO(PPE_MODEL_PATH)

print("Loading Fire/Smoke model...")
fire_model = YOLO(FIRE_SMOKE_MODEL_PATH)

print("\n================ PPE MODEL ================")
print(ppe_model.names)

print("\n============= FIRE/SMOKE MODEL ============")
print(fire_model.names)

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened")
    exit()

# ============================================================
# REAL-TIME DETECTION
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame")
        break

    # ========================================================
    # PPE DETECTION
    # ========================================================

    ppe_results = ppe_model.predict(
        source=frame,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    ppe_result = ppe_results[0]

    # ========================================================
    # FIRE / SMOKE DETECTION
    # ========================================================

    fire_results = fire_model.predict(
        source=frame,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    fire_result = fire_results[0]

    # ========================================================
    # DRAW PPE DETECTIONS
    # ========================================================

    annotated = ppe_result.plot()

    # ========================================================
    # FIRE / SMOKE DETECTION
    # ========================================================

    fire_detected = False
    smoke_detected = False

    if fire_result.boxes is not None:

        for box in fire_result.boxes:

            confidence = float(box.conf[0])

            # Ignore weak detections
            if confidence < 0.25:
                continue

            class_id = int(box.cls[0])

            class_name = fire_model.names[class_id]

            # Convert to lowercase so Fire/fire/Smoke/smoke
            # are handled consistently
            class_name_lower = class_name.lower()

            # Bounding box coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # =================================================
            # FIRE
            # =================================================

            if "fire" in class_name_lower:

                fire_detected = True

                label = f"FIRE {confidence:.2f}"

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1, max(y1 - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                print(
                    f"FIRE DETECTED | Confidence: {confidence:.2f}"
                )

            # =================================================
            # SMOKE
            # =================================================

            elif "smoke" in class_name_lower:

                smoke_detected = True

                label = f"SMOKE {confidence:.2f}"

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    3
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1, max(y1 - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )

                print(
                    f"SMOKE DETECTED | Confidence: {confidence:.2f}"
                )

    # ========================================================
    # STATUS MESSAGE
    # ========================================================

    if fire_detected and smoke_detected:

        status = "!!! FIRE + SMOKE DETECTED !!!"

        cv2.putText(
            annotated,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

    elif fire_detected:

        status = "!!! FIRE DETECTED !!!"

        cv2.putText(
            annotated,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

    elif smoke_detected:

        status = "!!! SMOKE DETECTED !!!"

        cv2.putText(
            annotated,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            3
        )

    else:

        status = "No Fire / Smoke"

        cv2.putText(
            annotated,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "FactoryGPT - PPE + Fire + Smoke Detection",
        annotated
    )

    # ========================================================
    # PRESS Q TO EXIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()