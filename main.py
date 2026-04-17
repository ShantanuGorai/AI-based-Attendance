import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def capture_face():

    stu_name = input('Enter student name : ')

    dataset_path = f"stu_dataset/{stu_name}"
    os.makedirs(dataset_path, exist_ok = True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for(x, y, w, h) in faces:
            count = count + 1
            face = gray[y : y + h, x : x + w]

            cv2.imwrite(f"{dataset_path}/{count}.jpg", face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Capturing Faces', frame)

        if count >= 50:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    print(f"Dataset for {stu_name} created successfully!!")
    print('Listing the student. . . ')
    model_training()
    print(f"{stu_name} listed succesfully")
    ch = int(input('1. Want to capture more faces ?\n2. Start attendace system\nEnter your choice : '))
    if (ch == 1):
        return
    elif (ch == 2):
        recognition()
    

def model_training():

    names = []
    label = []
    face = []
    student_id = 0

    path_data = 'stu_dataset'

    for student_name in os.listdir(path_data):
        student_name_path = os.path.join(path_data, student_name)

        if not os.path.isdir(student_name_path):
            continue

        names.append(student_name)

        for img_name in os.listdir(student_name_path):
            image_path = os.path.join(student_name_path, img_name)

            image = cv2.imread(image_path, 0)
            image = cv2.resize(image, (100, 100))
            face.append(image.flatten())
            label.append(student_id)

        student_id = student_id + 1

    faces = np.array(face)
    label = np.array(label)

    model = KNeighborsClassifier(n_neighbors = 3)
    model.fit(faces, label)

    with open('face_model.pkl', 'wb') as f:
        pickle.dump((model, names), f)

    print(f"Name : {names}")
    print("Labels : ", set(label))
    return

    
def recognition():
    attendace = {}
    data = []

    frame_count = 0
    attention_counter = {}


    if not os.path.exists('face_model.pkl'):
        print("Model not found! Training now...")
        model_training()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    with open('face_model.pkl', 'rb') as f:
        model, names = pickle.load(f)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y : y + h, x : x + w]
            face = cv2.resize(face, (100, 100)).flatten()

            prediction = model.predict([face])[0]
            distances, indices = model.kneighbors([face])
            if distances[0][0] > 5000:
                name = 'Unknown'
            else:
                name = names[prediction]
            
            if name != "Unknown":
                if name not in attention_counter:
                    attention_counter[name] = 0
                attention_counter[name] += 1


            current_time = datetime.now()
            if name != 'Unknown':
                if name not in attendace:
                    attendace[name] = {'entry':current_time, 'exit':None}
                else:
                    attendace[name]['exit'] = current_time

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('detecting', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    attention_scores = {}
    for name, count in attention_counter.items():
        score = (count / frame_count) * 100
        attention_scores[name] = round(score, 2)

    for name, times in attendace.items():
        entry = times['entry']
        exit = times['exit']

        if exit is None:
            duration = 0
        else:
            duration = (exit - entry).seconds / 60

        if (duration >= 35):
            status = 'Present'
        else:
            status = 'Absent'

        attention = attention_scores.get(name, 0)
        entry_str = entry.strftime("%H:%M:%S")
        exit_str = exit.strftime("%H:%M:%S") if exit else "N/A"
        data.append([name, entry_str, exit_str, round(duration,2), status, attention])
    
    df = pd.DataFrame(data, columns = ['Name', 'Entry', 'Exit', 'Duration (in minutes)', 'Status', 'attention'])
    df.to_csv("attendance.csv", index=False, sep = ',')
    print("Attendance saved!")
    print("Attention Scores:", attention_scores)
    print("\n ATTENDANCE REPORT \n")
    print(df.to_string(index=False))

def graph_presentastion():
    df = pd.read_csv('attendance.csv')

    names = df['Name']
    duration = df['Duration (in minutes)']

    plt.figure()
    plt.bar(names, duration, color = '#BFBFBF')
    plt.xlabel('Student Names')
    plt.ylabel('Duration')
    plt.title('Student vs Duration')
    plt.xticks(rotation=30)
    plt.show()

    present_count = (df["Status"] == "Present").sum()
    absent_count = (df["Status"] == "Absent").sum()

    plt.figure()
    plt.bar(["Present", "Absent"], [present_count, absent_count])
    plt.title("Attendance Summary")
    plt.ylabel("Number of Students")
    plt.show()

    attention = df["attention"]

    plt.figure()
    plt.bar(names, attention)
    plt.title("Student vs Attention Score")
    plt.xlabel("Students")
    plt.ylabel("Attention (%)")
    plt.xticks(rotation=30)
    plt.show()

while True:
    choice = int(input('1. For captiring a face\n2. Start attendace system\n3. To show graph representation\n4. Exit\nEnter yout choice : '))
    match choice:
        case 1:
            capture_face()
            break
        case 2:
            recognition()
            break

        case 3:
            graph_presentastion()
            break

        case 4:
            print('Exited . . .')
            break
        case _ :
            print('Enter a valide choice !')

        


        