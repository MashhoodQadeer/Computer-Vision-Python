from pathlib import Path
from ultralytics import YOLO

dataFile = Path("./data.yaml").resolve()
model = YOLO("yolov8n.pt")
results = model.train(data=str(dataFile), epochs=1)