import numpy as np
import cv2
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json


# Load the FaceNet model

def load_saved_model():
    '''Load the saved model from the disk'''
    try:
        with open('keras-facenet-h5/model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        
        # Load the model architecture from the JSON
        model = model_from_json(loaded_model_json)
        
        # Load the weights into the model
        model.load_weights('keras-facenet-h5/model.h5')
        
        return model
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Use the function
model = load_saved_model()
if model:
    print("Model loaded successfully!")
else:
    print("Failed to load the model.")


# model_path = "path_to_your_facenet_model.h5"
# model = load_model(model_path)

root_folder_path = 'processed_images'

def preprocess_image(img):
    # Resize image to 160x160
    img = cv2.resize(img, (160, 160))
    # Normalize pixel values to [-1, 1]
    img = img.astype('float32') / 127.5 - 1
    return img

def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    # make prediction to get embedding
    yhat = model.predict(np.expand_dims(face_pixels, axis=0))
    return yhat[0]

embeddings = []
labels = []

# Iterate through each folder and process images
for folder_name in os.listdir(root_folder_path):
    folder_path = os.path.join(root_folder_path, folder_name)
    
    if os.path.isdir(folder_path):
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            embeddings.append(get_embedding(model, preprocess_image(image)))
            labels.append(folder_name)

# Store the embeddings and labels to a file
data = {
    "embeddings": embeddings,
    "labels": labels
}
print(data['labels'])
print(data['embeddings'])

filename = "embeddings_and_labels.pkl"
with open(filename, 'wb') as f:
    pickle.dump(data, f)

print(f"Embeddings and labels saved to {filename}")
