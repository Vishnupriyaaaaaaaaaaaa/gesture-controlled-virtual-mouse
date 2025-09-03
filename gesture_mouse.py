import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Button, Controller
import ctypes
import time

class HandGestureMouseController:
    def __init__(self, screen_width, screen_height, smoothing=5):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.smoothing = smoothing
        self.prev_x, self.prev_y = 0, 0
        self.curr_x, self.curr_y = 0, 0
        self.mouse = Controller()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.last_left_click_time = 0
        self.last_right_click_time = 0
        self.click_delay = 0.5  

    def finger_states(self, hand_landmarks):
        tips_ids = [4, 8, 12, 16, 20]
        states = []

        if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
            states.append(True)
        else:
            states.append(False)

        for id in range(1, 5):
            tip_y = hand_landmarks.landmark[tips_ids[id]].y
            pip_y = hand_landmarks.landmark[tips_ids[id] - 2].y
            states.append(tip_y < pip_y)

        return states

    def detect_gesture(self, hand_landmarks):
        fingers = self.finger_states(hand_landmarks)

        if fingers == [False, False, False, False, False]:
            return 'fist'
        if fingers == [True, True, True, True, True]:
            return 'palm'
        if fingers == [True, False, False, False, False]:
            return 'thumbs_up' 
        if fingers == [False, True, True, False, False]:
            return 'peace'
        if fingers == [False, False, True, True, False]:
            return 'peace_inverted'
        if fingers == [False, True, True, True, False]:
            return 'three2'
        return 'unknown'

    def move_mouse(self, x, y):
        self.curr_x = self.prev_x + (x - self.prev_x) / self.smoothing
        self.curr_y = self.prev_y + (y - self.prev_y) / self.smoothing

        self.mouse.position = (int(self.curr_x), int(self.curr_y))

        self.prev_x, self.prev_y = self.curr_x, self.curr_y

    def click_left(self):
        now = time.time()
        if now - self.last_left_click_time > self.click_delay:
            self.mouse.click(Button.left, 1)
            self.last_left_click_time = now

    def click_right(self):
        now = time.time()
        if now - self.last_right_click_time > self.click_delay:
            self.mouse.click(Button.right, 1)
            self.last_right_click_time = now

    def scroll_up(self):
        self.mouse.scroll(0, 2)

    def scroll_down(self):
        self.mouse.scroll(0, -2)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                gesture = self.detect_gesture(hand_landmarks)

                index_tip = hand_landmarks.landmark[8]
                x = int(index_tip.x * self.screen_width)
                y = int(index_tip.y * self.screen_height)

                if gesture == 'palm':
                    self.move_mouse(x, y)
                    cv2.putText(frame, "Move Mouse", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                elif gesture == 'fist':
                    self.click_left()
                    cv2.putText(frame, "Left Click (Fist)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                elif gesture == 'thumbs_up':
                    self.click_right()
                    cv2.putText(frame, "Right Click (Thumbs Up)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                elif gesture == 'peace':
                    self.click_left()
                    self.click_left()  
                    cv2.putText(frame, "Double Click (Peace)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                elif gesture == 'peace_inverted':
                    self.scroll_up()
                    cv2.putText(frame, "Scroll Up (Peace Inverted)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                elif gesture == 'three2':
                    self.scroll_down()
                    cv2.putText(frame, "Scroll Down (Three2)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                else:
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

            else:
                cv2.putText(frame, "No Hand Detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Virtual Mouse", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    user32 = ctypes.windll.user32
    screen_w, screen_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    controller = HandGestureMouseController(screen_w, screen_h)
    controller.run()
