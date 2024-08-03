import os
import face_recognition
from PIL import Image

def crop_and_resize_face(image_path, output_path, size=(216, 216)):
    # Load the image with face_recognition
    image = face_recognition.load_image_file(image_path)

    # Find all face locations in the image
    face_locations = face_recognition.face_locations(image)

    # If no faces or more than one face is found, we'll skip the image
    if len(face_locations) != 1:
        print(f"Skipped {image_path} due to {'no' if len(face_locations) == 0 else 'multiple'} faces detected.")
        return

    top, right, bottom, left = face_locations[0]

    # Convert the image to a PIL format
    pil_image = Image.fromarray(image)
    
    # Crop the face out of the image
    face_image = pil_image.crop((left, top, right, bottom))
    
    # Resize the face image to the specified size
    face_image = face_image.resize(size)

    # Save the result
    face_image.save(output_path)

input_base_directory = "unprocessed_images"
output_base_directory = "processed_images"

# Using os.walk to traverse through subdirectories
for dirpath, dirnames, filenames in os.walk(input_base_directory):
    # Determine the output directory
    structure = os.path.join(output_base_directory, os.path.relpath(dirpath, input_base_directory))
    
    if not os.path.isdir(structure):
        os.makedirs(structure)

    # Loop over each image in the current directory
    for filename in filenames:
        if filename.endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(dirpath, filename)
            output_path = os.path.join(structure, filename)

            crop_and_resize_face(input_path, output_path)




