import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('C:/Users/Dell XPS White/Desktop/model.h5')

# Define gesture names
gesture_names = {
    0: 'Z', 1: 'R', 2: 'V', 3: 'F', 4: 'E', 5: 'T', 6: 'M', 7: 'K', 8: 'J', 9: 'A', 
    10: 'C', 11: 'H', 12: 'P', 13: 'G', 14: 'X', 15: 'I', 16: 'Q', 17: 'swaad', 18: 'B', 
    19: 'L', 20: 'U', 21: 'S', 22: 'N', 23: 'D', 24: 'O', 25: 'Y', 26: 'W', 27: 'aliph'
}

# Function to preprocess image
def preprocess_image(image):
    # Resize, convert to grayscale, and normalize the image
    image = cv2.resize(image, (100, 100))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.astype('float32') / 255.0
    # Add channel dimension
    gray_image = np.expand_dims(gray_image, axis=-1)
    return gray_image

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Failed to open webcam.")
else:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Preprocess frame
        preprocessed_frame = preprocess_image(frame)

        # Make prediction
        prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0))[0]
        predicted_label = np.argmax(prediction)
        gesture_name = gesture_names.get(predicted_label, "Unknown")

        # Display the predicted gesture name on the frame
        cv2.putText(frame, f"Predicted Gesture: {gesture_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Real-Time Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()
