# File: main.py

import cv2
import face_recognition
import os
import pickle
import time
import datetime

# Paths
FACE_DB = "face_database/"
LOG_FILE = "logs/entry_exit_log.csv"
ALERTS_FOLDER = "alerts/"
ENCODINGS_FILE = "models/face_encodings.pkl"

# Load known face encodings
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as file:
        known_faces, known_names = pickle.load(file)
else:
    known_faces, known_names = [], []

# Initialize video capture (CCTV or webcam)
cap = cv2.VideoCapture(0)  # Change 0 to CCTV IP stream if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # Log entry/exit
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a") as log:
            log.write(f"{name},{timestamp}\n")

        # Save unknown faces
        if name == "Unknown":
            filename = f"{ALERTS_FOLDER}{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)

        # Draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
