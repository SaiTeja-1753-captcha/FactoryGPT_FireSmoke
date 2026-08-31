from ultralytics import YOLO

model = YOLO("model/best.pt")

print("Model loaded successfully!")
print("Model classes:")
print(model.names)