from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # This will download the model if it's not cached
model.info()  # Show model info