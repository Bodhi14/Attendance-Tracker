import cv2
import pickle
import numpy as np
import os

def train(roll_no, n): 
        
    print("Training the model for : ", roll_no)
    Video = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    data = []

    i = 0

    while True:
        webval,frame = Video.read() 
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Faces = detector.detectMultiScale(gray_scale, 1.3, 4) 

        for (x,y,w,h) in Faces:
            cropped_image = frame[y:y+h, x:x+w, :]
            resized_image = cv2.resize(cropped_image, (50, 50))
            if len(data)<=100 and i%10 == 0:
                data.append(resized_image)
            i=i+1
            cv2.putText(frame, str(len(data)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (15,15,255), 1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), 4)
        
        cv2.imshow("frame", frame)
        k = cv2.waitKey(1) 
        if k == ord('e') or len(data)==100:
            break

    Video.release()
    cv2.destroyAllWindows()

    data = np.asarray(data)
    data = data.reshape(100, -1)

    if 'roll_nos.pkl' not in os.listdir('data/'):
        roll_nos = [roll_no]*100
        with open('data/roll_nos.pkl', 'wb') as f:
            pickle.dump(roll_nos, f)
    else:
        with open('data/roll_nos.pkl', 'rb') as f:
            roll_nos = pickle.load(f) 
        roll_nos = roll_nos + [roll_no]*100
        with open('data/roll_nos.pkl', 'wb') as f:
            pickle.dump(roll_nos, f)

    if 'faces_data.pkl' not in os.listdir('data/'):
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(data, f)
    else:
        with open('data/faces_data.pkl', 'rb') as f:
            images = pickle.load(f) 
            images = np.append(images, data, axis = 0)
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(roll_nos, f)









