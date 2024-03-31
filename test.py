from sklearn.neighbors import KNeighborsClassifier

import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

def test():
        detector = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

        with open('data/names.pkl', 'rb') as f:
                classes = pickle.load(f)

        with open('data/faces_data.pkl', 'rb') as f:
                faces = pickle.load(f)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces, classes)

        attr = ['Name', 'Date', 'Time']

        while True:
                Video = cv2.VideoCapture(0)
                webval,frame = Video.read() #provides 2 values -> webcam value(boolean) and the frame
                gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                Faces = detector.detectMultiScale(gray_scale, 1.3, 4) #width and height of the images

                for (x,y,w,h) in Faces:
                        cropped_image = frame[y:y+h, x:x+w, :]
                        resized_image = cv2.resize(cropped_image, (50, 50)).flatten().reshape(1, -1)
                        output = knn.predict(resized_image)
                        Time = time.time()
                        date = datetime.fromtimestamp(Time).strftime("%d-%m-%Y")
                        timestamp = datetime.fromtimestamp(Time).strftime("%H: %M: %S")
                        isExists = os.path.isfile("Records/Attendance_" + date + ".csv")
                        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                        cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), 4)
                        record = [str(output[0]), str(date), str(timestamp)]
                cv2.imshow("frame", frame)
                k = cv2.waitKey(1) #to break the loop

                Video.release()

                if k == ord('a'):
                        if isExists:
                                pass
                        else:
                                with open("Records/Attendance_" + date + ".csv", "+a") as sheet:
                                        wr = csv.writer(sheet)
                                        wr.writerow(attr)
                                        wr.writerow(record)
                        sheet.close()
                        break
                if k == ord('e'):
                        break

        cv2.destroyAllWindows()
