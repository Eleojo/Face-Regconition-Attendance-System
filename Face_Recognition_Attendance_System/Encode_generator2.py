
import face_recognition
import os
import pickle

path = 'processed_images'

encodeListKnown = []
studentIds = []

model_type = "cnn"  # You can switch between "hog" and "cnn" based on your needs and hardware capability

# Using os.walk to go through each file in the directory tree
for dirpath, dnames, fnames in os.walk(path):
    for fname in fnames:
        print(f"Processing {fname} ...")

        if fname.endswith((".jpg", ".jpeg", ".png")):  # Checking for relevant file types
            curImgPath = os.path.join(dirpath, fname)
            person = os.path.basename(dirpath)  # Assuming the name of person is the directory name
            curImg = face_recognition.load_image_file(curImgPath)
            
            # First, locate the face in the image
            face_locations = face_recognition.face_locations(curImg, model=model_type)
            
            if len(face_locations) > 0:
                # Encode the detected face
                encodings = face_recognition.face_encodings(curImg, face_locations)
                encode = encodings[0]
                encodeListKnown.append(encode)
                studentIds.append(person)
            else:
                print(f"No face detected in {fname}")

# Save the encodings and studentIds to a file
print(len(encodeListKnown), len(studentIds))
with open("encoded_faces2.pkl", "wb") as file:
    pickle.dump({"encodings": encodeListKnown, "studentIds": studentIds}, file)
