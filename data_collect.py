import cv2
import mediapipe as mp
import csv
import os

# mediapipe hands initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Directory for saving images
root_dir = "gesture_images"
os.makedirs(root_dir, exist_ok=True)

# Initialize Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("! Camera not initialized.")
    exit()

print("s: start recording data")
print("q: quit")

recording = False
gesture_label = None
image_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate bounding box around hand
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Expand bounding box slightly for better cropping
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Crop the hand region
            cropped_hand = frame[y_min:y_max, x_min:x_max]
                
            # Save the current frame as an image
            if recording and gesture_label:
                # Create a directory for the gesture label if it doesn't exist
                gesture_dir = os.path.join(root_dir, gesture_label)
                os.makedirs(gesture_dir, exist_ok=True)

                # Save the cropped hand image
                image_path = os.path.join(gesture_dir, f"{image_count}.jpg")
                cv2.imwrite(image_path, cropped_hand)
                image_count += 1


    # Display Instructions
    cv2.putText(frame, "Press 's' to start/stop recording.", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Recording: {'ON' if recording else 'OFF'}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if recording else (255, 0, 0), 2)
    cv2.putText(frame, "Press 'q' to quit.", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("***capture***", frame)

    # Handle key presses
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    elif key == ord('s'):
        recording = not recording  # Toggle recording state
        if recording:
            gesture_label = input("Enter the gesture label: ")  # Set label for recording
        else:
            gesture_label = None  # Clear label when recording stops

# Release resources
cap.release()
cv2.destroyAllWindows()
