import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import pyttsx3
import threading
import tensorflow as tf

# Function for text-to-speech synthesis
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set the width of the frames
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set the height of the frames
cap.set(cv2.CAP_PROP_FPS, 30)  # Set the frame rate

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
frame_skip = 5  # Process every 5th frame

labels = ["Dad", "Hello", "No", "Thank You", "Yes"]

# Load the trained model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(imgSize, imgSize, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])

# Load the trained weights
model.load_weights('model_weights.h5')

while True:
    for i in range(frame_skip):
        success, img = cap.read()

    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Check if the bounding box is within the image bounds
        if x-offset >= 0 and y-offset >= 0 and x+w+offset < img.shape[1] and y+h+offset < img.shape[0]:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            # Perform inference using the loaded model
            prediction = model.predict(np.expand_dims(imgWhite, axis=0))
            index = np.argmax(prediction)
            label_text = labels[index]

            # Speak out the recognized label using threading
            threading.Thread(target=speak, args=(label_text,)).start()

            cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x-offset+400, y-offset+60-50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, label_text, (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y+h + offset), (0, 255, 0), 4)

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
