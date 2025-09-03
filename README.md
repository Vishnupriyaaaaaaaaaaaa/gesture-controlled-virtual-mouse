# Hand Gesture Mouse Controller

A Python application that uses computer vision and hand gestures to control your mouse. This project leverages **MediaPipe** for hand tracking and **pynput** for controlling mouse actions such as moving the cursor, clicking, and scrolling using specific hand gestures.

---

## Features

- **Move Mouse:** Move your mouse pointer by showing an open palm.
- **Left Click:** Make a fist to perform a left click.
- **Right Click:** Show a thumbs up to perform a right click.
- **Double Click:** Show the "peace" gesture to double-click.
- **Scroll Up:** Show an inverted "peace" gesture to scroll up.
- **Scroll Down:** Show the "three2" gesture (three fingers) to scroll down.

---

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- MediaPipe
- pynput
- numpy

You can install the dependencies using pip:

pip install opencv-python mediapipe pynput numpy
Usage

Clone this repository or download the script.

Run the script:

python hand_gesture_mouse_controller.py


Allow access to your webcam.

Use the gestures in front of the webcam to control your mouse.

Press q to quit the application.
