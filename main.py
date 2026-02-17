import ultralytics
import supervision
import torch
import cv2
import numpy as np
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO
import winsound
import threading
import time

beep_active = False
sound_thread = None

def play_beep():
    global beep_active
    while beep_active:
        winsound.Beep(1000, 500)
        time.sleep(0.5)

def is_emergency_vehicle(frame, box):
    x, y, w, h = box
    x1 = int(max(0, x - w/2))
    y1 = int(max(0, y - h/2))
    x2 = int(min(frame.shape[1], x + w/2))
    y2 = int(min(frame.shape[0], y + h/2))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Red mask
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    # Blue mask
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    red_pixels = cv2.countNonZero(mask_red)
    blue_pixels = cv2.countNonZero(mask_blue)
    total_pixels = crop.shape[0] * crop.shape[1]
    if red_pixels > 0.005 * total_pixels and blue_pixels > 0.005 * total_pixels:
        return True
    return False

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Set up video capture
cap = cv2.VideoCapture("My movie 014.mp4")

# Define the line coordinates
START = sv.Point(182, 254)
END = sv.Point(462, 254)

# Store the track history
track_history = defaultdict(lambda: [])

# Create a dictionary to keep track of objects that have crossed the line
crossed_objects = {}

# Create a dictionary to keep track of emergency vehicles that have crossed the line
emergency_crossed = {}

# List to store confidence scores for accuracy calculation
confidence_scores = []

# Open a video sink for the output video
video_info = sv.VideoInfo.from_video_path("My movie 014.mp4")
with sv.VideoSink("output_emergency.mp4", video_info) as sink:
    
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, classes=[2, 3, 5, 7], persist=True, save=True, tracker="bytetrack.yaml")

            # Get the boxes and track IDs
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xywh.cpu()
                confs = results[0].boxes.conf.cpu().tolist()
                confidence_scores.extend(confs)
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                else:
                    track_ids = []
            else:
                boxes = []
                track_ids = []

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Initialize signal state
            signal_state = "normal"

            # Plot the tracks and count objects crossing the line
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)

                is_emergency = is_emergency_vehicle(frame, box)

                if is_emergency:
                    # Check if emergency vehicle crosses the line
                    if START.x < x < END.x and abs(y - START.y) < 5:
                        emergency_crossed[track_id] = True
                    else:
                        # Emergency vehicle not crossed, set signal to emergency
                        if track_id not in emergency_crossed:
                            signal_state = "emergency"
                    # Annotate emergency vehicle with red box and label
                    cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, "Ambulance", (int(x - w / 2), int(y - h / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    # Normal vehicle
                    if START.x < x < END.x and abs(y - START.y) < 5:
                        if track_id not in crossed_objects:
                            crossed_objects[track_id] = True
                        # Annotate normal vehicle as it crosses the line
                        cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

            # Determine signal color
            if signal_state == "emergency":
                signal_color = (0, 255, 0)  # green
            else:
                signal_color = (0, 0, 255)  # red

            # Control alert sound
            if signal_state == "emergency" and not beep_active:
                beep_active = True
                if sound_thread is None or not sound_thread.is_alive():
                    sound_thread = threading.Thread(target=play_beep)
                    sound_thread.start()
            elif signal_state != "emergency" and beep_active:
                beep_active = False

            # Draw the line on the frame with signal color
            cv2.line(annotated_frame, (START.x, START.y), (END.x, END.y), signal_color, 2)

            # Draw signal circle
            cv2.circle(annotated_frame, (video_info.width - 50, 50), 30, signal_color, -1)
            cv2.putText(annotated_frame, signal_state.upper(), (video_info.width - 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Write the count of objects on each frame
            count_text = f"Objects crossed: {len(crossed_objects)}"
            cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Write the frame with annotations to the output video
            sink.write_frame(annotated_frame)
        else:
            break

# Release the video capture
cap.release()

# Calculate and print average confidence score
if confidence_scores:
    avg_conf = sum(confidence_scores) / len(confidence_scores)
    print(f"Average confidence score: {avg_conf:.4f}")
else:
    print("No detections made.")
