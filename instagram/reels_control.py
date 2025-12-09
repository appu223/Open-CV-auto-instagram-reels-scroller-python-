import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

gesture_delay = 1
last_gesture_time = 0

def detect_swipe(lms):
    y_positions = [lm[1] for lm in lms]
    return sum(y_positions) / len(y_positions)


cap = cv2.VideoCapture(0)
previous_avg_y = None

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            handLms = result.multi_hand_landmarks[0]

            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            avg_y = detect_swipe(lm_list)
            current_time = time.time()

            if previous_avg_y is not None and current_time - last_gesture_time > gesture_delay:
                movement = avg_y - previous_avg_y

                if movement < -25:
                    pyautogui.press("up")
                    cv2.putText(frame, "Previous Reel", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                    last_gesture_time = current_time

                elif movement > 25:
                    pyautogui.press("down")
                    cv2.putText(frame, "Next Reel", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                    last_gesture_time = current_time

            previous_avg_y = avg_y

        cv2.imshow("Gesture Controller", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
