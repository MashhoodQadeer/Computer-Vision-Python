from ultralytics import YOLO
import cv2

model = YOLO("./monkey_detect/yolov8n_custom2/weights/best.pt")
print(model.names)

input_path = "./videos/input.mp4"
output_path = "./videos/output.mp4"

cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=frame, conf=0.25, verbose=False)

    annotated_frame = results[0].plot()

    out.write(annotated_frame)
    cv2.imshow("Monkey Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
