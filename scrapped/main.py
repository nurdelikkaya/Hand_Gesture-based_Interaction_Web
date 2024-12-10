import mediapipe as mp
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

app = FastAPI()

# Serve static files (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Button state
class ButtonState(BaseModel):
    row: int
    col: int
    button: str

grid_size = 3
buttons = [f"Button {i + 1}" for i in range(grid_size * grid_size)]

# Initialize button position and state
current_position = (0, 0)  # Start with Button 1 (top-left corner)
current_button = {
    "row": 0,
    "col": 0,
    "button": buttons[0]  # Button 1
}

# Gesture Recognizer Configuration
gesture_model_path = "custom_gestures.task"  
base_options = python.BaseOptions(model_asset_path=gesture_model_path)
gesture_recognizer_options = vision.GestureRecognizerOptions(base_options=base_options)
gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_recognizer_options)

@app.get("/")
async def serve_frontend():
    """Serve the main HTML page."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_path, "static", "index.html")
    with open(html_path, "r") as f:
        return HTMLResponse(f.read())

@app.get("/current_button")
def get_current_button():
    """Get the currently highlighted button."""
    return current_button

@app.post("/process_frame/")
async def process_frame(file: UploadFile = File(...)):
    """Process an image frame to detect gestures."""
    global current_position, current_button
    # Read the uploaded image
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Perform gesture recognition
    gesture_results = gesture_recognizer.recognize(mp_image)

    # Gesture handling
    if gesture_results.gestures:
        top_gesture = gesture_results.gestures[0][0]  # Top gesture
        gesture_name = top_gesture.category_name

        row, col = current_position

        # Map gestures to button movements
        if gesture_name == "Victory" and col < grid_size - 1:  # Move right
            current_position = (row, col + 1)
        elif gesture_name == "Pointing_Up" and col > 0:  # Move left
            current_position = (row, col - 1)
        elif gesture_name == "Thumb_Up" and row > 0:  # Move up
            current_position = (row - 1, col)
        elif gesture_name == "Thumb_Down" and row < grid_size - 1:  # Move down
            current_position = (row + 1, col)

        # Update the currently highlighted button
        button_index = current_position[0] * grid_size + current_position[1]
        current_button = {
            "row": current_position[0],
            "col": current_position[1],
            "button": buttons[button_index],
        }

    return {"status": "success", "current_button": current_button}
