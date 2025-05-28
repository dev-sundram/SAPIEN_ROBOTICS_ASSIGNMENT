# import cv2
# import time
# from ultralytics import YOLO

# # Load the YOLOv8 pretrained model
# model = YOLO("yolov8n.pt")  # You can also use 'yolov8s.pt' or others

# # Initialize webcam (0 is default camera)
# cap = cv2.VideoCapture(0)

# # Set frame size (Optional)
# cap.set(3, 640)  # Width
# cap.set(4, 480)  # Height

# # FPS tracking
# prev_time = 0

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break

#     # Inference
#     results = model(frame, stream=True)

#     # Loop through detections
#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             # Get coordinates
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = float(box.conf[0])
#             cls = int(box.cls[0])
#             label = model.names[cls]

#             # Draw rectangle and label
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#     # Calculate and display FPS
#     curr_time = time.time()
#     fps = 1 / (curr_time - prev_time)
#     prev_time = curr_time
#     cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     # Show frame
#     cv2.imshow("YOLOv8 Real-time Detection", frame)

#     # Break on 'q' key
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()


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
