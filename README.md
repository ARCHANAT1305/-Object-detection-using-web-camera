# Object-detection-using-web-camera
### AIM:
To perform real-time object detection using a trained YOLO v4 model through your laptop camera.
### ALGORITHM:
STEP 1: Import required libraries.  
STEP 2: Load YOLOv4 model and config files.  
STEP 3: Get output layer names.  
STEP 4: Load class labels.  
STEP 5: Start webcam.  
STEP 6: Capture frames in a loop.   
STEP 7: Convert frame to blob and set input.  
STEP 8: Run forward pass for detections.  
STEP 9: Filter high-confidence objects.  
STEP 10: Draw boxes and labels.  
STEP 11: Show output frame.  
STEP 12: Press ‘q’ to quit and close windows.  
### PROGRAM:
```
import cv2
import numpy as np
weights_path = "yolov4.weights"
config_path = "yolov4.cfg"
names_path = "coco.names"
import os
for f in [weights_path, config_path, names_path]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing file: {f}")
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
classes = open(names_path).read().strip().split("\n")
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x, center_y, w, h = map(int, detection[:4] * [width, height, width, height])
                x, y = center_x - w // 2, center_y - h // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{classes[class_id]} {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv4 Real-Time Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```
### OUTPUT:
![WhatsApp Image 2025-10-08 at 09 41 33_fc0a5907](https://github.com/user-attachments/assets/d707f274-6143-475d-b0fe-dde9dc8cc575)
### RESULT :

Thus the program to perform real-time object detection using a trained YOLO v4 model was executed successfully.
