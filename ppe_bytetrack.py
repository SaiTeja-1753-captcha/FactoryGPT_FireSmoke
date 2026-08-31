from ultralytics import YOLO

# Load PPE model
model = YOLO(r"model\best.pt")

# Video source
# 0 = laptop webcam
# Or replace with a video filename
source = 0

results = model.track(
    source=source,
    tracker="bytetrack.yaml",
    conf=0.25,
    persist=True,
    show=True
)