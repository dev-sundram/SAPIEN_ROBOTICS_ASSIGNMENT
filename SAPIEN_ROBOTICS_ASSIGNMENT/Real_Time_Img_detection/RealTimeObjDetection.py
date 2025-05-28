import cv2
import time
from ultralytics import YOLO

# Load the pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30.0
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('demo_output.mp4', fourcc, fps, (frame_width, frame_height))

# Start time
start_time = time.time()
frame_count = 0
classes_detected = set()

print("Recording started. Press 'q' to quit early.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    annotated_frame = results.plot()

    # Collect class names
    for result in results.boxes.cls:
        class_id = int(result)
        class_name = model.names[class_id]
        classes_detected.add(class_name)

    # Show and write the video
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)
    out.write(annotated_frame)

    frame_count += 1
    elapsed_time = time.time() - start_time

    if cv2.waitKey(1) & 0xFF == ord('q') or elapsed_time > 60:
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()

# Performance summary
avg_fps = frame_count / elapsed_time
print(f"\n📝 Recording complete.\n➡️ Duration: {elapsed_time:.2f} seconds\n➡️ Avg FPS: {avg_fps:.2f}")
print(f"➡️ Classes Detected: {sorted(classes_detected)}")
