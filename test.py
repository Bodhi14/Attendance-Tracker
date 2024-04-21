from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

def test(duration=2):
    detector = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    with open('data/roll_nos.pkl', 'rb') as f:
        classes = pickle.load(f)

    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, classes)

    Video = cv2.VideoCapture(0)
    start_time = time.time()

    while True:
        webval, frame = Video.read()  
        if frame is None:
            break
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = detector.detectMultiScale(gray_scale, 1.3, 4)

        detected_labels = []
        detected_images = []

        for (x, y, w, h) in detected_faces:
            cropped_image = frame[y:y+h, x:x+w, :]
            resized_image = cv2.resize(cropped_image, (50, 50)).flatten().reshape(1, -1)
            detected_images.append(resized_image)

        if detected_images:
            detected_images = np.vstack(detected_images)
            detected_labels = knn.predict(detected_images)

        for (x, y, w, h), label in zip(detected_faces, detected_labels):
            current_time = time.strftime("%H:%M:%S")
            current_date = time.strftime("%d-%m-%Y")
            student_label = str(label)

            attendance_file = f"Records/{student_label}_Attendance.csv"
            is_exists = os.path.isfile(attendance_file)

            with open(attendance_file, 'a') as sheet:
                wr = csv.writer(sheet)
                if not is_exists:
                    wr.writerow(['Roll_No', 'Date', 'Time'])
                wr.writerow([student_label, current_date, current_time])

            cv2.putText(frame, student_label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 4)

        # cv2.imshow("frame", frame)
        k = cv2.waitKey(1) 

        if time.time() - start_time >= duration:
            break
        Video.release()
        
    cv2.destroyAllWindows()
