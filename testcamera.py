import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av
from ultralytics import YOLO  # Importing YOLOv8

# Load YOLOv8 model
model = YOLO('./ColorShapeSize.pt')


# Define your function to detect and draw bounding boxes
def detect_and_draw_boxes(image):
    # Run YOLO object detection on the image/frame
    results = model(image)

    # Extract bounding boxes and labels, and draw on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            label = result.names[int(box.cls[0])]  # Object label

            # Draw bounding box and label on the frame
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return image


# WebRTC video frame callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Process frame for object detection
    img_with_boxes = detect_and_draw_boxes(img)

    return av.VideoFrame.from_ndarray(img_with_boxes, format="bgr24")


# Streamlit app UI
st.title("Live Object Detection with YOLOv8")

# WebRTC streamer for live camera feed
webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
