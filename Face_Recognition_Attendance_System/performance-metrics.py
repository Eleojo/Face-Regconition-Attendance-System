import os
import face_recognition
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Prepare Your Data
# Organize your dataset into three separate folders, each containing images of a specific person.

# Step 2: Training
# Train your dlib model with your dataset (not included in this code).

# Step 3: Testing
# Define the path to the folder containing test images
test_folder = 'processed_images'

# Lists to store predicted identities and ground truth labels
predicted_identities = []
ground_truth = []

# Load your trained dlib model here (replace with your actual model)
# For example, you can load the model using the following:
# trained_model = dlib.load("your_trained_model.dat")

# Iterate through the test images
for person_folder in os.listdir(test_folder):
    person_path = os.path.join(test_folder, person_folder)
    
    # Extract the person's name from the folder name
    person_name = person_folder
    
    for image_file in os.listdir(person_path):
        test_image_path = os.path.join(person_path, image_file)
        test_image = face_recognition.load_image_file(test_image_path)
        
        # Perform face recognition using your trained dlib model
        # Replace this with your actual inference code
        face_encodings = face_recognition.face_encodings(test_image)
        
        if len(face_encodings) > 0:
            # For simplicity, assume only one face is detected in each image
            encoding = face_encodings[0]
            
            # Assign a label (identity) to the face (replace with actual recognition logic)
            # For this example, we assume the folder name matches the identity
            predicted_identity = person_name
        else:
            # If no face is detected, mark the prediction as "Unknown"
            predicted_identity = "Unknown"
        
        # Append the predicted identity and ground truth to the respective lists
        predicted_identities.append(predicted_identity)
        ground_truth.append(person_name)

# Step 4: Calculate Performance Metrics
# Calculate Accuracy
accuracy = accuracy_score(ground_truth, predicted_identities)
print(f"Accuracy: {accuracy:.2f}")

# Calculate Precision
precision = precision_score(ground_truth, predicted_identities, labels=list(set(ground_truth)), average="micro")
print(f"Precision: {precision:.2f}")

# Calculate Recall
recall = recall_score(ground_truth, predicted_identities, labels=list(set(ground_truth)), average="micro")
print(f"Recall: {recall:.2f}")

# Calculate F1-Score
f1 = f1_score(ground_truth, predicted_identities, labels=list(set(ground_truth)), average="micro")
print(f"F1-Score: {f1:.2f}")


conf_matrix = confusion_matrix(ground_truth, predicted_identities)

# Get a list of unique labels (classes)
labels = list(set(ground_truth))

# Plot the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()