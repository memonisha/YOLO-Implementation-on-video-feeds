import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Make sure these paths are correct
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO names for object detection (or use your own labels)
with open("coco.names", "r") as f:  # Ensure coco.names file is in the correct path
    classes = [line.strip() for line in f.readlines()]

# Initialize VideoCapture
cap = cv2.VideoCapture("carvideo.mp4")  # Replace with your video path

# Set the frame resolution and FPS if necessary
cap.set(cv2.CAP_PROP_FPS, 30)  # Set the frame rate to 30 FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set the width of the frame
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set the height of the frame

# Define frame skip rate for performance (skip every nth frame)
frame_skip = 5
frame_counter = 0

# Frame processing
while True:
    ret, frame = cap.read()
    if not ret:
        break  # If the video ends, break the loop

    if frame_counter % frame_skip == 0:
        # Resize frame to YOLO input size (320x320 or 416x416 for better accuracy)
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Parse YOLO outputs
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:  # Confidence threshold for detecting objects
                    # Get the object's coordinates and dimensions
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

        # Draw bounding boxes and labels
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)  # Color of bounding box

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show the frame
        cv2.imshow("Object Detection", frame)

    frame_counter += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
