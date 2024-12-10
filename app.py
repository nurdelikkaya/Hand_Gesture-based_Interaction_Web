import tkinter as tk
from tkinter import filedialog
import pyautogui
import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import mediapipe as mp
import threading
from math import sqrt
from PIL import Image, ImageTk, ImageDraw

# Path to your trained .task file for gesture recognition
task_file = "custom_gestures.task"

# Gesture Recognizer Configuration
base_options = BaseOptions(model_asset_path=task_file)
gesture_recognizer_options = vision.GestureRecognizerOptions(base_options=base_options)
gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_recognizer_options)

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Setting the scroll start detected variables as global :")
up_start_detected = False
down_start_detected = False

# Distance function
def distance(x1, y1, x2, y2):
    return int(sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


# Tkinter Interface
class GestureReaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture-Based Reader")
        self.root.attributes('-fullscreen', True)  # Open in full-screen mode

        # Default font size
        self.font_size = 18

        # Frame for Text Widget and Scrollbar
        self.text_frame = tk.Frame(root)
        self.text_frame.pack(padx=20, pady=10, expand=True, fill="both")

        # Scrollable Text Widget
        self.scrollbar = tk.Scrollbar(self.text_frame, orient="vertical")
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_area = tk.Text(
            self.text_frame,
            wrap="word",
            font=("Arial", self.font_size),
            yscrollcommand=self.scrollbar.set,
            width=100,  # Increased width
            height=10,  # Increased height
            padx=10,
            pady=10
        )
        self.text_area.pack(side=tk.LEFT, fill="both", expand=True)
        self.scrollbar.config(command=self.text_area.yview)

        # Button Frame
        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        # Load File Button
        self.load_button = self.create_button("Load File", self.load_file, 0)

        # Increase Font Size Button
        self.increase_font_button = self.create_button("Increase Text Size", self.increase_font, 1)

        # Decrease Font Size Button
        self.decrease_font_button = self.create_button("Decrease Text Size", self.decrease_font, 2)

        # Exit Button
        self.exit_button = self.create_button("Exit", self.on_close, 3)

        # Gesture Display Label
        self.current_gesture = tk.StringVar(value="No Gesture Detected")
        self.gesture_label = tk.Label(root, textvariable=self.current_gesture, font=("Arial", 14))
        self.gesture_label.pack(pady=5)

        # Small Camera Canvas
        self.camera_frame = tk.Frame(root)
        self.camera_frame.pack(side=tk.LEFT, padx=10, pady=10, anchor="sw")

        self.canvas = tk.Canvas(self.camera_frame, width=200, height=150)
        self.canvas.pack()

        # Webcam Thread
        self.running = True
        self.cap = None  # Initialize OpenCV video capture object
        self.webcam_thread = threading.Thread(target=self.process_webcam)
        self.webcam_thread.daemon = True  # Ensure thread stops with the app
        self.webcam_thread.start()

        # State to prevent click spamming
        self.click_detected = False

    def create_button(self, text, command, col):
        """Create a button with hover and click feedback."""
        button = tk.Button(
            self.button_frame,
            text=text,
            font=("Arial", 16),
            width=15,
            height=2,
            relief="raised",
            command=command
        )
        button.grid(row=0, column=col, padx=10, pady=5)

        # Bindings for hover, click, and release states
        button.bind("<Enter>", lambda e: button.config(bg="lightblue"))  # Hover
        button.bind("<Leave>", lambda e: button.config(bg="SystemButtonFace"))  # Default
        button.bind("<ButtonPress>", lambda e: button.config(bg="blue"))  # Clicked
        button.bind("<ButtonRelease>", lambda e: button.config(bg="lightgreen"))  # Released

        return button


    def load_file(self):
        """Load a text file into the text widget."""
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    self.text_area.delete(1.0, tk.END)
                    self.text_area.insert(tk.END, file.read())  
            except Exception as e:
                print(f"Error loading file: {e}")

    def increase_font(self):
        """Increase the font size of the text."""
        self.font_size += 2
        self.text_area.config(font=("Arial", self.font_size))

    def decrease_font(self):
        """Decrease the font size of the text."""
        if self.font_size > 8:
            self.font_size -= 2
            self.text_area.config(font=("Arial", self.font_size))

    def process_webcam(self):
        """Process gestures and display the camera feed on the canvas."""
        global up_start_detected, down_start_detected  # Declare global variables

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Webcam not initialized.")
            return

        hands = mp.solutions.hands.Hands()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Prepare the frame for Mediapipe
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = hands.process(rgb_frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = gesture_recognizer.recognize(mp_image)
            if result.gestures:
                    gesture = result.gestures[0][0].category_name

            if output.multi_hand_landmarks:
                for hand in output.multi_hand_landmarks:
                    hand_landmarks = hand.landmark
                    wrist_x = wrist_y = index_x = index_y = middle_x = middle_y = 0

                    for i, landmark in enumerate(hand_landmarks):
                        x = int(landmark.x * frame_width)
                        y = int(landmark.y * frame_height)

                        if i == 0:  # Wrist
                            wrist_x, wrist_y = x, y
                        if i == 8:  # Index tip
                            index_x, index_y = x, y
                        if i == 12:  # Middle finger tip
                            middle_x, middle_y = x, y

                    # Calculate distances for open palm detection
                    index_distance = distance(wrist_x, wrist_y, index_x, index_y)
                    middle_distance = distance(wrist_x, wrist_y, middle_x, middle_y)

                    # Cursor movement with an open palm
                    if gesture == "open_palm":
                        palm_x = (wrist_x + index_x + middle_x) // 3
                        palm_y = (wrist_y + index_y + middle_y) // 3
                        screen_x = int((screen_width / frame_width) * palm_x)
                        screen_y = int((screen_height / frame_height) * palm_y)
                        pyautogui.moveTo(screen_x, screen_y)
                        cursor_position = (palm_x, palm_y)
                        self.current_gesture.set("Cursor Movement")

                    # Simulate a click with pinch gesture (index tip close to middle tip)
                    if gesture == "pinch":
                        if not self.click_detected:
                            pyautogui.click()
                            self.click_detected = True
                            self.current_gesture.set("Click")
                    else:
                        self.click_detected = False
                    
                    if gesture == "scrollUP_start":
                        up_start_detected = True
                        self.current_gesture.set("Scroll Up Start")

                    if gesture == "scrollUP_end" and up_start_detected:
                        self.text_area.yview_scroll(-3, "units")  # Scroll up 3 units
                        up_start_detected = False
                        self.current_gesture.set("Scroll Up End")

                    if gesture == "scrollDOWN_start":
                        down_start_detected = True
                        self.current_gesture.set("Scroll Down Start")

                    if gesture == "scrollDOWN_end" and down_start_detected:
                        self.text_area.yview_scroll(3, "units")  # Scroll down 3 units
                        down_start_detected = False
                        self.current_gesture.set("Scroll Down End")

                    if gesture == "scrollRight":
                        self.text_area.xview_scroll(3, "units")  # Scroll right 3 units
                        self.current_gesture.set("Scroll to Right")

                    if gesture == "scrollLeft":
                        self.text_area.xview_scroll(-3, "units")  # Scroll left 3 units
                        self.current_gesture.set("Scroll to Left")

                    if gesture == "idle":
                        self.current_gesture.set("Idle")
              


            # Resize and display the camera feed
            resized_frame = cv2.resize(rgb_frame, (200, 150))
            img = Image.fromarray(resized_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk  # Keep reference to avoid garbage collection

        self.cap.release()
        cv2.destroyAllWindows()

    def on_close(self):
        """Handle cleanup and close the application."""
        self.running = False  # Stop the webcam thread loop
        if self.cap:
            self.cap.release()  # Release the camera resource
        if self.webcam_thread.is_alive():
            self.webcam_thread.join()  # Wait for the webcam thread to finish
        cv2.destroyAllWindows()  # Ensure all OpenCV windows are closed
        self.root.destroy()  # Close the Tkinter window
# Run the App
if __name__ == "__main__":
    root = tk.Tk()
    app = GestureReaderApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
