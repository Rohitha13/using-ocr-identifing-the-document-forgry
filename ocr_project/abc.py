import cv2
import streamlit as st
import numpy as np

st.title("Suspicious Word Detector")

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Load the input image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to the grayscale image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours and check for inconsistencies in color
    for contour in contours:
        # Get the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Check if the aspect ratio of the bounding box is appropriate for a word
        aspect_ratio = float(w) / h
        if aspect_ratio > 1.5 and aspect_ratio < 10:
            # Extract the region of interest (ROI) using the bounding box
            roi = image[y:y+h, x:x+w]

            # Convert the ROI to grayscale and apply thresholding
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            # Calculate the mean color of the ROI
            mean_color = cv2.mean(roi_thresh)

            # Check if the mean color is too light or too dark
            if mean_color[0] < 100 or mean_color[0] > 200:
                # Draw a red bounding box around the suspicious word
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display the image with suspicious words highlighted
    st.image(image, caption='Processed Image', use_column_width=True)
