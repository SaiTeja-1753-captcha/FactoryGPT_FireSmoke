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
    verbose=False
)


# =========================================================
# RUN FIRE/SMOKE MODEL
# =========================================================

print("Running Fire/Smoke model...")

firesmoke_results = firesmoke_model.predict(
    source=image,
    conf=0.25,
    verbose=False
)


# =========================================================
# DRAW PPE DETECTIONS
# =========================================================

output = image.copy()

print("\n========== PPE DETECTIONS ==========")

for result in ppe_results:

    for box in result.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        class_name = ppe_model.names[class_id]

        print(
            f"PPE: {class_name} "
            f"| Confidence: {confidence:.2f}"
        )

        # Draw bounding box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
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
# DRAW FIRE/SMOKE DETECTIONS
# =========================================================

print("\n========== FIRE/SMOKE DETECTIONS ==========")

for result in firesmoke_results:

    for box in result.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        class_name = firesmoke_model.names[class_id]

        print(
            f"Fire/Smoke: {class_name} "
            f"| Confidence: {confidence:.2f}"
        )

        # Draw bounding box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        # Label
        label = f"Fire/Smoke: {class_name} {confidence:.2f}"

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )


# =========================================================
# SAVE RESULT
# =========================================================

output_folder = Path("IntegratedResults")
output_folder.mkdir(exist_ok=True)

output_path = output_folder / "integrated_result.jpg"

cv2.imwrite(str(output_path), output)


# =========================================================
# FINAL STATUS
# =========================================================

print("\n======================================")
print("      INTEGRATED DETECTION COMPLETE")
print("======================================")

print(f"\nResult saved at:")
print(output_path)

cv2.imshow("FactoryGPT Integrated Detection", output)

print("\nPress any key on the image window to close.")

cv2.waitKey(0)
cv2.destroyAllWindows()