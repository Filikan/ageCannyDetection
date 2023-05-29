import cv2
import numpy as np

low_threshold = 30
high_threshold = 150

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def calculate_wrinkles(image):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return np.mean(edges)


def predict_age(image):
    faces = detect_faces(image)
    if len(faces) == 0:
        return 'Face could not be found.'

    max_wrinkleness = 0
    for (x, y, w, h) in faces:
        face_image = image[y:y + h, x:x + w]
        wrinkleness = calculate_wrinkles(face_image)
        if wrinkleness > max_wrinkleness:
            max_wrinkleness = wrinkleness

    if max_wrinkleness > 50:
        return 'Old'
    elif max_wrinkleness > 30:
        return 'Middle-aged'
    else:
        return 'Young'


image = cv2.imread('') #It should be your image path.

predicted_age = predict_age(image)

print('Predicted Age Group:', predicted_age)

