import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyautogui

class HandGestureVirtualMouse:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        
        self.gestures = ['fist', 'ok', 'palm', 'peace', 'peace_inverted', 'three2']
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.screen_width, self.screen_height = pyautogui.size()
        
        self.prev_x, self.prev_y = 0, 0
        self.smoothing = 3
        
        self.current_action = "No Action"
        
        self.scrolling = False
    
    def extract_landmarks(self, landmarks, max_landmarks=21):
        landmark_list = [[lm.x, lm.y] for lm in landmarks.landmark]
        
        if len(landmark_list) > max_landmarks:
            landmark_list = landmark_list[:max_landmarks]
        elif len(landmark_list) < max_landmarks:
            landmark_list += [[0, 0]] * (max_landmarks - len(landmark_list))
        
        flat_landmarks = [coord for point in landmark_list for coord in point]
        
        return np.array(flat_landmarks).reshape(1, -1, 1)
    
    def control_mouse(self, hand_landmarks):
        index_finger = hand_landmarks.landmark[8]
        
        x = int(index_finger.x * self.screen_width)
        y = int(index_finger.y * self.screen_height)
        
        x = int(self.prev_x + (x - self.prev_x) / self.smoothing)
        y = int(self.prev_y + (y - self.prev_y) / self.smoothing)
        
        pyautogui.moveTo(x, y)
        
        self.prev_x, self.prev_y = x, y
        
        self.current_action = "Moving Cursor"
    
    def recognize_gesture_and_control(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(rgb_frame)
        
        self.current_action = "No Hand Detected"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                processed_landmarks = self.extract_landmarks(hand_landmarks)
                
                prediction = self.model.predict(processed_landmarks)
                gesture_index = np.argmax(prediction)
                
                if self.gestures[gesture_index] == 'palm':
                    self.control_mouse(hand_landmarks)
                    self.scrolling = False
                elif self.gestures[gesture_index] == 'three2':
                    pyautogui.rightClick()
                    self.current_action = "Right Click"
                    self.scrolling = False
                elif self.gestures[gesture_index] == 'ok':
                    pyautogui.click()
                    self.current_action = "Left Click"
                    self.scrolling = False
                elif self.gestures[gesture_index] == 'peace':
                    pyautogui.scroll(100)
                    self.current_action = "Scroll Up"
                    self.scrolling = True
                elif self.gestures[gesture_index] == 'peace_inverted':
                    pyautogui.scroll(-100)
                    self.current_action = "Scroll Down"
                    self.scrolling = True
                elif self.gestures[gesture_index] == 'fist':
                    self.current_action = "Stop Scrolling"
                    self.scrolling = False
        
        cv2.putText(
            frame, 
            self.current_action, 
            (10, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        return frame

def main():
    model_path = 'model.h5'
    
    virtual_mouse = HandGestureVirtualMouse(model_path)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    pyautogui.FAILSAFE = False
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            frame = cv2.flip(frame, 1)
            
            processed_frame = virtual_mouse.recognize_gesture_and_control(frame)
            
            cv2.imshow('Hand Gesture Virtual Mouse', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()