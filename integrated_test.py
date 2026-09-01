from ultralytics import YOLO
from pathlib import Path
import cv2

# =========================================================
# LOAD BOTH MODELS
# =========================================================

ppe_model = YOLO(r"model\best.pt")
firesmoke_model = YOLO(r"model\FireSmoke_best.pt")

print("PPE classes:", ppe_model.names)
print("Fire/Smoke classes:", firesmoke_model.names)


# =========================================================
# INPUT IMAGE
# =========================================================

image_path = input("\nEnter image path: ").strip().strip('"')

image = cv2.imread(image_path)

if image is None:
    print("\nERROR: Image could not be opened.")
    print("Check the image path and filename.")
    exit()


# =========================================================
# RUN PPE MODEL
# =========================================================

print("\nRunning PPE model...")

ppe_results = ppe_model.predict(
    source=image,
    conf=0.25,
    imgsz=640,
    verbose=False
)


# =========================================================
# RUN FIRE/SMOKE MODEL
# =========================================================

print("Running Fire/Smoke model...")

firesmoke_results = firesmoke_model.predict(
    source=image,
    conf=0.25,
    imgsz=640,
    verbose=False
)


# =========================================================
# CREATE OUTPUT IMAGE
# =========================================================

output = image.copy()


# =========================================================
# DRAW PPE DETECTIONS
# =========================================================

print("\n========== PPE DETECTIONS ==========")

for result in ppe_results:

    if result.boxes is None:
        continue

    for box in result.boxes:

        confidence = float(box.conf[0])

        if confidence < 0.25:
            continue

        class_id = int(box.cls[0])
        class_name = ppe_model.names[class_id]

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        print(
            f"PPE: {class_name} | "
            f"Confidence: {confidence:.2f}"
        )

        # -------------------------------------------------
        # PPE = GREEN
        # OpenCV uses BGR format
        # (0, 255, 0) = GREEN
        # -------------------------------------------------

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"PPE: {class_name} {confidence:.2f}"

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


# =========================================================
# FIRE / SMOKE DETECTIONS
# =========================================================

print("\n========== FIRE / SMOKE DETECTIONS ==========")

fire_detected = False
smoke_detected = False

for result in firesmoke_results:

    if result.boxes is None:
        continue

    for box in result.boxes:

        confidence = float(box.conf[0])

        if confidence < 0.25:
            continue

        class_id = int(box.cls[0])

        class_name = firesmoke_model.names[class_id]

        # Convert class name to lowercase
        class_name_lower = class_name.lower().strip()

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        # =================================================
        # FIRE
        # =================================================

        if "fire" in class_name_lower:

            fire_detected = True

            print(
                f"FIRE: Confidence = {confidence:.2f}"
            )

            # GREEN bounding box
            # BGR = (0, 255, 0)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            label = f"FIRE {confidence:.2f}"

            cv2.putText(
                output,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        # =================================================
        # SMOKE
        # =================================================

        elif "smoke" in class_name_lower:

            smoke_detected = True

            print(
                f"SMOKE: Confidence = {confidence:.2f}"
            )

            # RED bounding box
            # BGR = (0, 0, 255)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            label = f"SMOKE {confidence:.2f}"

            cv2.putText(
                output,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


        # =================================================
        # UNKNOWN CLASS
        # =================================================

        else:

            print(
                f"Unknown Fire/Smoke class: "
                f"{class_name} | "
                f"Confidence: {confidence:.2f}"
            )


# =========================================================
# FINAL STATUS
# =========================================================

print("\n========== FINAL STATUS ==========")

if fire_detected and smoke_detected:

    status = "!!! FIRE + SMOKE DETECTED !!!"
    status_color = (0, 0, 255)

elif fire_detected:

    status = "!!! FIRE DETECTED !!!"
    status_color = (0, 255, 0)

elif smoke_detected:

    status = "!!! SMOKE DETECTED !!!"
    status_color = (0, 0, 255)

else:

    status = "NO FIRE / SMOKE DETECTED"
    status_color = (255, 255, 255)


print(status)

cv2.putText(
    output,
    status,
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    status_color,
    3
)


# =========================================================
# SAVE RESULT
# =========================================================

output_folder = Path("IntegratedResults")
output_folder.mkdir(exist_ok=True)

output_path = output_folder / "integrated_result.jpg"

success = cv2.imwrite(
    str(output_path),
    output
)

if success:
    print("\n======================================")
    print("   INTEGRATED DETECTION COMPLETE")
    print("======================================")
    print(f"\nResult saved at:")
    print(output_path)
else:
    print("\nERROR: Could not save output image.")


# =========================================================
# DISPLAY RESULT
# =========================================================

cv2.imshow(
    "FactoryGPT - PPE + Fire + Smoke Detection",
    output
)

print("\nPress any key on the image window to close.")

cv2.waitKey(0)
cv2.destroyAllWindows()
