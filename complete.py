import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import numpy as np
from PIL import Image

# Load YOLO model (YOLOv8)
model = YOLO('./FinalTrained.pt')  # Update with your trained model path

# Set up the Streamlit page
st.title("YOLO Object Detection")
st.text("Select input method to perform object detection.")

# Option to choose input format
input_option = st.radio("Choose input format", ("Upload Image/Video", "Live Camera", "Capture from Camera"))

if input_option == "Upload Image/Video":
    upload_option = st.radio("Upload Image or Video", ("Image", "Video"))

    if upload_option == "Image":
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            # Load the image
            image = Image.open(uploaded_image)
            frame = np.array(image)

            # Run YOLO object detection on the uploaded image
            results = model(frame)

            # Extract bounding boxes and labels and draw on the frame
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    conf = box.conf[0]  # Confidence score
                    label = result.names[int(box.cls[0])]  # Object label

                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Convert to RGB for display in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb)

    elif upload_option == "Video":
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)

            # Streamlit placeholder for video
            frame_placeholder = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Run YOLO object detection on the frame
                results = model(frame)

                # Draw bounding boxes on the frame
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0]
                        label = result.names[int(box.cls[0])]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Display the frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb)

            cap.release()

elif input_option == "Live Camera":
    # Live camera feed option
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture image")
                break

            # Run YOLO object detection on the frame
            results = model(frame)

            # Extract bounding boxes and labels and draw on the frame
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    conf = box.conf[0]  # Confidence score
                    label = result.names[int(box.cls[0])]  # Object label

                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Convert BGR to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb)

        cap.release()

elif input_option == "Capture from Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        # Convert the uploaded camera image to numpy array
        frame = np.array(Image.open(camera_image))

        # Run YOLO object detection on the captured image
        results = model(frame)

        # Extract bounding boxes and labels and draw on the frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0]  # Confidence score
                label = result.names[int(box.cls[0])]  # Object label

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Convert to RGB for display in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb)
