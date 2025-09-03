import cv2
import mediapipe as mp
import json
import os
from datetime import datetime

class GestureDataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.current_gesture = "none"
        self.collected_data = {}
        self.sample_count = 0
        
    def collect_data(self, gesture_name):
        """Collect gesture data"""
        cap = cv2.VideoCapture(0)
        
        print(f"Collecting data for gesture: {gesture_name}")
        print("Press SPACE to capture, 'q' to quit, 'n' for next gesture")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract landmarks
                    landmarks = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
                    
                    # Display info
                    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Samples: {self.sample_count}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "Press SPACE to capture", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and results.multi_hand_landmarks:
                # Save landmark data
                timestamp = str(datetime.now().timestamp())
                self.collected_data[f"{gesture_name}_{timestamp}"] = {
                    "landmarks": [landmarks]
                }
                self.sample_count += 1
                print(f"Captured sample {self.sample_count} for {gesture_name}")
                
            elif key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

def main():
    collector = GestureDataCollector()
    gestures = ['palm', 'ok', 'fist', 'peace', 'peace_inverted', 'three2']
    
    os.makedirs('../datasets', exist_ok=True)
    
    for gesture in gestures:
        collector.sample_count = 0
        collector.collected_data = {}
        
        print(f"\n--- Collecting data for {gesture} ---")
        input("Press Enter to start collecting...")
        
        collector.collect_data(gesture)
        
        # Save data
        filename = f"../datasets/{gesture}.json"
        with open(filename, 'w') as f:
            json.dump(collector.collected_data, f, indent=2)
            
        print(f"Saved {collector.sample_count} samples to {filename}")

if __name__ == '__main__':
    main()