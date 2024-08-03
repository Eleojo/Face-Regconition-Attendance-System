from mtcnn.mtcnn import MTCNN
import cv2

# Load image
filename = "321654.png"
pixels = cv2.imread(filename)

# Create the detector using default weights
detector = MTCNN()

# Detect faces in the image
faces = detector.detect_faces(pixels)

# Print face locations
for face in faces:
    print(face['box'])


# from mtcnn.mtcnn import MTCNN
# import cv2

# # Load image
# filename = "thumbnail.png"
# pixels = cv2.imread(filename)

# # Create the detector using default weights
# detector = MTCNN()

# # Detect faces in the image
# faces = detector.detect_faces(pixels)

# # Draw bounding boxes on the image
# for face in faces:
#     x, y, width, height = face['box']
#     cv2.rectangle(pixels, (x, y), (x+width, y+height), (0, 255, 0), 2)

# # Display the image with bounding boxes
# cv2.imshow('Face Detection', pixels)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
