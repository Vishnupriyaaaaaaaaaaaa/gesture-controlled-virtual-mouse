import cv2
import mediapipe as mp
import json
import os

GESTURE_NAME = "three2"  # Change this per gesture you want to collect
SAVE_PATH = f"datasets/{GESTURE_NAME}.json"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

dataset = {}
count = 0

if not os.path.exists("datasets"):
    os.makedirs("datasets")

print(f"Recording gesture: {GESTURE_NAME}")
print(" Press 's' to save frame, 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Gesture: {GESTURE_NAME} | Samples: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Collector", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s') and results.multi_hand_landmarks:
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.append([lm.x, lm.y])

        dataset[f"{GESTURE_NAME}_{count}"] = {"landmarks": [landmarks]}
        count += 1

with open(SAVE_PATH, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"\n Saved {count} samples to: {SAVE_PATH}")
cap.release()
cv2.destroyAllWindows()
