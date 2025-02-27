import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize video capture and hand detector
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y = 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                # Draw hexagon pattern on each landmark
                hexagon_points = []
                for i in range(6):
                    angle = 2 * np.pi / 6 * i
                    point_x = x + 10 * np.cos(angle)
                    point_y = y + 10 * np.sin(angle)
                    hexagon_points.append((int(point_x), int(point_y)))
                hexagon_points = np.array(hexagon_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [hexagon_points], isClosed=True, color=(255, 0, 0), thickness=2)

                # Draw circle on index finger (ID 8)
                if id == 8:
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y

                # Draw circle on thumb (ID 4)
                if id == 4:
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y

                    # Calculate the distance between index and thumb
                    if abs(index_y - thumb_y) < 20:
                        pyautogui.click()
                        pyautogui.sleep(1)
                    elif abs(index_y - thumb_y) < 100:
                        pyautogui.moveTo(index_x, index_y)

            # Draw an arrow from the wrist (ID 0) to the index finger (ID 8)
            wrist = landmarks[0]
            index_finger = landmarks[8]
            wrist_x = int(wrist.x * frame_width)
            wrist_y = int(wrist.y * frame_height)
            index_finger_x = int(index_finger.x * frame_width)
            index_finger_y = int(index_finger.y * frame_height)
            cv2.arrowedLine(frame, (wrist_x, wrist_y), (index_finger_x, index_finger_y), (173, 216, 230), 3) # Light blue color

    cv2.imshow('Virtual Mouse', frame)
    cv2.waitKey(1)
