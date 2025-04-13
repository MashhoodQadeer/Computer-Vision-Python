from pathlib import Path
from ultralytics import YOLO

# Load data and model
dataFile = Path("./data.yaml").resolve()
model = YOLO("yolov8n.pt")  # start from pre-trained YOLOv8n

# Train the model
results = model.train(
    data=str(dataFile),
    epochs=100,              # <-- increase epochs
    imgsz=640,               # image size
    batch=16,                # adjust for your GPU/CPU
    patience=10,             # early stop if no improvement
    save=True,               # save best model
    project="monkey_detect",# folder to save results
    name="yolov8n_custom"    # subfolder name
)
