import cv2
import numpy as np
from playsound import playsound
import threading

# Load Haar Cascade Classifiers for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

# Variables to store previous positions for movement calculation
prev_face_center = None
face_not_detected_counter = 0
alert_threshold = 30  # Number of frames with no face detection before alerting

def play_alert():
   
    playsound('C:\\Users\\DELL 7490\\Downloads\\alert_sound.mp3')

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)
    # Convert the BGR image to grayscale for Haar Cascade detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        face_not_detected_counter = 0  # Reset counter when a face is detected
        
        for (x, y, w, h) in faces:
            # Calculate the center of the face
            face_center = (x + w // 2, y + h // 2)
            
            # Draw rectangle around the face
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Draw a circle on the face center
            cv2.circle(image, face_center, 3, (255, 0, 0), -1)

            # Detect eyes within the face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Draw rectangles around detected eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Check if the face is within the camera's frame bounds
            if face_center[0] < 50 or face_center[0] > image.shape[1] - 50 or face_center[1] < 50 or face_center[1] > image.shape[0] - 50:
                # Face is out of bounds, play an alert
                threading.Thread(target=play_alert).start()

            # Update previous face center for movement tracking (optional)
            prev_face_center = face_center
    else:
        # Increment counter if no face is detected
        face_not_detected_counter += 1

        if face_not_detected_counter > alert_threshold:
            # Play alert if the face hasn't been detected for enough frames
            threading.Thread(target=play_alert).start()
            face_not_detected_counter = 0  # Reset after playing alert
    
    # Display the image
    cv2.imshow('Face and Eye Movement Detection with Alerts', image)

    # Break loop with 'q' key
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
