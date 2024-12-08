import cv2
import mediapipe as mp
import pyautogui as agui
import os
from math import * 

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # suppress other cpu usage

# mediapipe hands initialization
capture_hands = mp.solutions.hands.Hands()
drawing_opt = mp.solutions.drawing_utils

scrWidth, scrHeight = agui.size()

camera = cv2.VideoCapture(0) # start camera
if not camera.isOpened():
    print("Error: Camera not initialized.")
    exit()

print("Camera started successfully.")

distance = lambda x1, y1, x2, y2 : int(sqrt((x2-x1)**2 +(y2-y1)**2))

while True: # continue until escape
    ret, frame = camera.read()
    if not camera.isOpened():
        print("! Camera feed lost.")
        break
    if not ret:
        print("! Unable to read frame from the camera.")
        continue
    # print("Frame captured successfully.")

    frame = cv2.flip(frame, 1)
    frHeight, frWidth, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    output = capture_hands.process(rgb_frame)
    hands = output.multi_hand_landmarks # if we do not only use 1 hand

    if hands:
        for h in hands: # processing left and right hand seperately
            drawing_opt.draw_landmarks(frame, h, mp.solutions.hands.HAND_CONNECTIONS)
            handLandmark = h.landmark
            wx = wy = tx = ty = xx = yy = ix = iy = mx = my = rx = ry = px = py = 0
            for num, lm in enumerate(handLandmark):
                # calculate x and y coordinates wrt frame size
                x = int(lm.x * frWidth) 
                y = int(lm.y * frHeight)
                if num == 0: # wrist 
                    wx, wy = x, y
                if num == 4: # thumb tip
                    tx, ty = x, y
                if num == 5: # index starting 
                    xx, yy = x, y
                if num == 8: # index tip
                    ix, iy = x, y
                if num == 12: # middle tip
                    mx, my = x, y
                if num == 16: # ring finger tip
                    rx, ry = x, y
                if num == 20: # pinky tip
                    px, py = x, y

                # calculations for open hand
                thumbDistance = distance(wx, wy, tx, ty)
                indexDistance = distance(wx, wy, ix, iy)
                midDist = distance(wx, wy, mx, my)
                ringDist = distance(wx, wy, rx, ry)
                pinkyDist = distance(wx, wy, px, py)
            print(f"Distances between wrist and each fingertip: {thumbDistance}\t{indexDistance}\t{midDist}\t{ringDist}\t{pinkyDist}\t")
            
            # if palm is open, move the cursor
            if thumbDistance >= 140 and indexDistance >= 160 and midDist >= 120 and ringDist >= 120 and pinkyDist >= 120: # if palm open
                mouseX = int((scrWidth / frWidth) * ((xx+wx)/2)) # calculate where middle of palm corresponds to 
                mouseY = int((scrHeight / frHeight) * ((yy+wy)/2)) # in x and y axis of the screen
                agui.moveTo(mouseX, mouseY) 
                
            if distance(ix, iy, tx, ty) < 50 and iy <= ty:  # If thumb and index are close
                print("Pinch!")
                agui.click()  # Simulate a click

    cv2.imshow("capture", frame) # shows the video capture from the camera
    key = cv2.waitKey(1)
    if key == 27: # if esc clicked
        break
camera.release()
cv2.destroyAllWindows()

