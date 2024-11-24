import streamlit as st
import cv2
import numpy as np

# Load Haar cascades for face and eye detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Streamlit app title
st.title("Real-time Face and Eye Detection")

# Add instructions to the Streamlit app
st.write("This app detects faces and eyes in real-time using your webcam. Click the button below to start the camera feed.")

# Button to start webcam
start_button = st.button("Start Webcam")

# Create a placeholder for the video
video_placeholder = st.empty()

if start_button:
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        st.error("Could not open webcam")
    
    while True:
        ret, img = cap.read()
        if not ret:
            break

        # Convert the image to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100))

        # Draw rectangle around faces and eyes
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Region of interest (ROI) for eyes within the detected face
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Detect eyes within the face ROI
            eyes = eye_classifier.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.putText(roi_color, "Eye", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert the image to RGB (Streamlit expects RGB format, not BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display the image in Streamlit
        video_placeholder.image(img_rgb, channels="RGB", use_column_width=True)

        # Break the loop if the user clicks 'Stop Webcam'
        stop_button = st.button("Stop Webcam")
        if stop_button:
            cap.release()
            video_placeholder.empty()
            break

    # Release the capture when done
    cap.release()

else:
    st.write("Click on 'Start Webcam' to begin face and eye detection.")

