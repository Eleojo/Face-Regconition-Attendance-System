import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
import firebase_admin
from firebase_admin import credentials, firestore, db, storage
import pandas as pd
import datetime
import json
import logging


#Load credentials
# cred = credentials.Certificate("faceattendance-dd5b9-firebase-adminsdk-g2czp-f9ea6e24ee.json")
# firebase_admin.initialize_app(cred, {
#     #'databaseURL': "https://faceattendancesystem-4df3a-default-rtdb.firebaseio.com/",
#     'storageBucket': "faceattendance-dd5b9.appspot.com"
# })

# Load Firebase credentials from environment variable
# cred_path = os.environ.get('FIREBASE_CREDENTIALS_PATH')
# if not cred_path:
#     print("Please set the FIREBASE_CREDENTIALS_PATH environment variable!")
#     exit()

# # Initialize Firebase Admin SDK
try:
    cred = credentials.Certificate("faceattendance-dd5b9-firebase-adminsdk-g2czp-f9ea6e24ee.json")
    firebase_admin.initialize_app(cred,{
        'storageBucket': "faceattendance-dd5b9.appspot.com"
    })
except Exception as e:
    print(f"Failed to initialize Firebase: {e}")
    exit()

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Request CourseCode and lecturer name
course_code = input('Enter course code : ')
#   get date
date = datetime.datetime.now().strftime('%Y-%m-%d')
current_datetime = datetime.datetime.now()
date_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

#Load students.json
with open('students.json','r') as file:
    students = json.load(file)

# Load course_info.json
with open('courses_info.json','r') as file:
    course_info = json.load(file)

# Load encodings and names
print("Loading Encode File ...")
with open("encoded_faces2.pkl", "rb") as file:
    data = pickle.load(file)
    encodeListKnown = data["encodings"]
    studentIds = data["studentIds"]
print("Encode File Loaded")
print(f"Number of faces encoded: {len(encodeListKnown)}")


# Reference Firestore
db = firestore.client()

TOLERANCE = 0.45

def speak_windows(text):
    command = f'powershell Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\\"{text}\\")'
    os.system(command)

#function to check if document exists
def document_exists(doc_ref):
    """Check if the document exists."""
    doc = doc_ref.get()
    return doc.exists

def load_img(image_path):
    img = cv2.imread(image_path)
    return img

bucket = storage.bucket()

cap = cv2.VideoCapture(2)
cap.set(3, 640)
cap.set(4, 480)


attendance = pd.DataFrame(columns=['Reg.No', 'Time','Date', 'Lecturer Name'])
imgBackground = cv2.imread('Resources/background.png')

# Importing the mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
# print(len(imgModeList))

# Load the encoding file


modeType = 0
counter = 0
id = -1
imgStudent = []

# url = "http://192.168.43.78/"


while True:
    success, img = cap.read()
    # img_resp = urllib.request.urlopen(url)
    # imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    # img =cv2.imdecode.findPosition(img)

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace, TOLERANCE)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print("matches", matches)
            # print("faceDis", faceDis)
            try:
                matchIndex = np.argmin(faceDis)
            except ValueError as e:
                pass
            # print("Match Index", matchIndex)
            if not any(matches):  # If no matches found for this face.
                cvzone.putTextRect(imgBackground, "Unknown", (275, 400))
                speak_windows("Sorry you didnt register this course")
                continue  # Skip the rest of the current loop iteration.

            elif matches[matchIndex]:
                print(f"this smallest face distance is {faceDis[matchIndex]}")
                attendance_time = datetime.datetime.now().strftime('%I:%M:%S %p')
                # print("Known Face Detected")
                # print(studentIds[matchIndex])
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                id = studentIds[matchIndex]
                if counter == 0:
                    cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                    cv2.imshow("Face Attendance", imgBackground)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
                    counter = 1
                    modeType = 1

        if counter != 0:

            if counter == 1:
                # create course for that date
                course_code_ref = db.collection('COURSES').document(course_code)
                # course_info = course_code_ref.get()
                # create student for that date
                student_info = course_code_ref.collection('ATTENDANCE').document(id)
                
                #   take attendance on excel sheet
                try:
                    attendance.loc[len(attendance)] = [id, attendance_time, date, course_info[course_code]['lecturer_name']]
                except KeyError:
                    print('course not registered')
                    # speak_windows(f" sorry {course_info[course_code]['lecturer_name']} {course_code} is not a registered course for this semester")
                # print("Excel sheet updated")
                # attendance.loc[len(attendance)] = [name, current_time,current_date, lecturer_name]

                if not document_exists(course_code_ref):
                   try:
                        course_code_ref.set({
                
                            'course_code': course_code,
                            'course_title': course_info[course_code]['title'],
                            'course_lecturer' : course_info[course_code]['lecturer_name'],
                            'date_created': date,
                    
                        })
                        print(f" {course_code} added on {date}.")
                   except Exception as e:
                       print(f"Failed to set COURSE document: {e}")
                else:
                    print(f"COURSE  {course_code} already exists on {date}.")



               # Check if student document exists
                if not document_exists(student_info):
                    try:
                        student_info.set({
                            
                            'id': students[id]['reg_no'],
                            'student_name' : students[id]['name'],
                            'student_id': id,
                            'time': attendance_time,
                            'date': date,
                            'gender' : students[id]['gender'],
                            'major': students[id]['major'],
                            'date_time': date_time,

                        })
                        print(f"Student {id} added to course {course_code} on {date}.")
                    except Exception as e:
                        print(f"Failed to set student document: {e}")
                else:
                    print(f"Student with reg number {id} already exists for course {course_code} on {date}.")

                # Get the Image from the storage
                # blob = bucket.get_blob(f'Images/{id}.png')
                # array = np.frombuffer(blob.download_as_string(), np.uint8)
                # imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

                # image_path = f"Images/{id}.png"
                # imgStudent = cv2.imread(image_path)

                # get image from firestore and cache it in imgStudent
                image_path = f"Images/{id}.png"
                # imgStudent = download_and_decode_image(bucket, image_path)
                imgStudent = load_img(image_path)

                # calculating elapsed time from time attendance was taken
                doc = student_info.get()
                datetimeObject = doc.to_dict().get('time')
                
                datetimeObject = datetime.datetime.strptime(datetimeObject, '%I:%M:%S %p')
                datetimeObject = datetime.datetime.combine(datetime.date.today(), datetimeObject.time())
                secondsElapsed = (datetime.datetime.now() - datetimeObject).total_seconds()
                print(secondsElapsed)
                print(secondsElapsed)
                if secondsElapsed > 50:
                    # student_info.update({
                    #     'num_of_attendance' : firestore.Increment(1)
                    # })
                    pass
                    
                else:
                    modeType = 3
                    counter = 0
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

            if modeType != 3:

                if 10 < counter < 20:
                    modeType = 2
                attendance_count = doc.to_dict().get('num_of_attendance')
                major = doc.to_dict().get('major')
                standing = doc.to_dict().get('standing')
                starting_year = doc.to_dict().get('starting_year')
                
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                if counter <= 10:
                    # if min(faceDis) < 0.6:
                    speak_windows(f"welcome {students[id]['name']}")
                    cv2.putText(imgBackground, str(attendance_count), (861, 125),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(major), (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(id), (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(standing), (910, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                # cv2.putText(imgBackground, str(student_info['year']), (1025, 625),
                    #           cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(starting_year), (1125, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                    (w, h), _ = cv2.getTextSize(students[id]['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (414 - w) // 2
                    cv2.putText(imgBackground, str(students[id]['name']), (808 + offset, 445),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)
                
                    imgBackground[175:175 + 216, 909:909 + 216] = imgStudent
                    # else:
                    #     cvzone.putTextRect(imgBackground, "Unknown", (275, 400))
                    #     speak_windows("sorry you did not register this course")

                counter += 1

                if counter >= 20:
                    counter = 0
                    modeType = 0
                    student_info = []
                    imgStudent = []
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    
    else:
        modeType = 0
        counter = 0
    # cv2.imshow("Webcam", img)
    
    
    cv2.imshow("Face Attendance", imgBackground)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

#   save attendance to excel sheet
try:
    with pd.ExcelWriter('attendance.xlsx', mode='a') as writer:  
        attendance_no_duplicates = attendance.drop_duplicates(subset='Reg.No', keep='first')
        attendance_no_duplicates.to_excel(writer, sheet_name=course_code, index=False)
    logger.info('Attendance Saved')
except PermissionError:
    logger.error("Permission error: The Excel file is already open. Please close it and try again.")
except ValueError:
    logger.error(f"{course_code} sheet already exists")
# try:
#     with pd.ExcelWriter('attendance.xlsx', mode='a') as writer:  
#         attendance.to_excel(writer, sheet_name=course_code, index=False)
#     logger.info('Attendance Saved')
# except PermissionError:
#     logger.error("Permission error: The Excel file is already open. Please close it and try again.")