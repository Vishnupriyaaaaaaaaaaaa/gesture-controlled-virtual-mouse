import cv2
import mediapipe as mp
from pynput.mouse import Button, Controller
import numpy as np

mouse = Controller()

import ctypes
user32 = ctypes.windll.user32
screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

mp_draw = mp.solutions.drawing_utils

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def finger_states(hand_landmarks):
    """
    Returns a list of finger states (True for extended, False for folded)
    Thumb: Compare tip and IP landmarks (x-axis)
    Other fingers: Compare tip and PIP landmarks (y-axis)
    """
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
def detect_gesture(hand_landmarks):
    """
    Detect gesture based on finger states and landmarks.
    Gestures:
      - 'fist': All fingers folded
      - 'palm': All fingers extended
      - 'ok': Thumb and index finger tips close, others folded
      - 'peace': Index and middle extended, others folded
      - 'peace_inverted': Middle and ring extended, others folded
      - 'three2': Index, middle, ring extended, others folded
    """

    fingers = finger_states(hand_landmarks)
    
    thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
    thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)

    if fingers == [False, False, False, False, False]:
        return 'fist'

   
    if fingers == [True, True, True, True, True]:
        return 'palm'

    
    if thumb_index_dist < 0.05 and fingers[2] == False and fingers[3] == False and fingers[4] == False:
        return 'ok'


    if fingers == [False, True, True, False, False]:
        return 'peace'

    
    if fingers == [False, False, True, True, False]:
        return 'peace_inverted'

    if fingers == [False, True, True, True, False]:
        return 'three2'

    return 'unknown'

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = detect_gesture(hand_landmarks)

          
            if gesture == 'palm':
                index_finger_tip = hand_landmarks.landmark[8]
                x = int(index_finger_tip.x * screen_width)
                y = int(index_finger_tip.y * screen_height)
                mouse.position = (x, y)
                cv2.putText(frame, "Move Mouse", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            elif gesture == 'fist':
                mouse.press(Button.left)
                mouse.release(Button.left)
                cv2.putText(frame, "Left Click (Fist)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            elif gesture == 'ok':
                mouse.press(Button.right)
                mouse.release(Button.right)
                cv2.putText(frame, "Right Click (OK)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            elif gesture == 'peace':
                mouse.click(Button.left, 2)  # double click
                cv2.putText(frame, "Double Click (Peace)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

            elif gesture == 'peace_inverted':
                # Scroll up as example
                mouse.scroll(0, 2)
                cv2.putText(frame, "Scroll Up (Peace Inverted)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            elif gesture == 'three2':
                # Scroll down as example
                mouse.scroll(0, -2)
                cv2.putText(frame, "Scroll Down (Three2)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

            else:
                cv2.putText(frame, f"Gesture: {gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

        else:
            cv2.putText(frame, "No Hand Detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Virtual Mouse", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

