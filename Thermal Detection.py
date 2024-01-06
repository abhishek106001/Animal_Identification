import numpy as np
import cv2

# Replace this with the correct camera index or device name
camera_index = 0
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a colormap to the grayscale image to simulate thermal colors
    thermal_frame = cv2.applyColorMap(gray_frame, cv2.COLORMAP_JET)

    # Display the pseudo-thermal image
    cv2.imshow("Pseudo-Thermal Image", thermal_frame)

    if cv2.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
