import pyautogui
import cv2
import mediapipe as mp
import os
import math
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -------------------- CONFIGURATION --------------------

STABLE_FRAMES = 8
CAMERA_INDEX = 0
THUMB_THRESHOLD = 0.50

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "assets", "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# -------------------- DISTANCE CALCULATION --------------------

def distance(point_a, point_b):
    return math.sqrt(
        (point_a.x - point_b.x) ** 2
        + (point_a.y - point_b.y) ** 2
    )


# -------------------- FINGER DETECTION --------------------

def is_finger_open(hand, tip, pip):
    return hand[tip].y < hand[pip].y


def get_finger_states(hand):
    # Order: Thumb, Index, Middle, Ring, Pinky

    palm_width = distance(hand[5], hand[17])

    if palm_width > 0:
        thumb_ratio = distance(hand[4], hand[5]) / palm_width
    else:
        thumb_ratio = 0

    thumb_open = thumb_ratio > THUMB_THRESHOLD

    index_open = is_finger_open(hand, 8, 6)
    middle_open = is_finger_open(hand, 12, 10)
    ring_open = is_finger_open(hand, 16, 14)
    pinky_open = is_finger_open(hand, 20, 18)

    finger_states = (
        int(thumb_open),
        int(index_open),
        int(middle_open),
        int(ring_open),
        int(pinky_open)
    )

    return finger_states, thumb_ratio


# -------------------- GESTURE RECOGNITION --------------------

def recognize_gesture(finger_states):

    if finger_states == (1, 1, 1, 1, 1):
        return "ACTIVATE"

    elif finger_states == (0, 1, 0, 0, 0):
        return "NEXT"

    elif finger_states == (0, 1, 1, 0, 0):
        return "PREVIOUS"

    elif finger_states == (1, 1, 0, 0, 1):
        return "SPIDEY"

    elif finger_states == (0, 0, 0, 0, 0):
        return "DEACTIVATE"

    return "UNKNOWN"


# -------------------- START MEDIAPIPE --------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Hand landmark model not found:\n{MODEL_PATH}\n"
        "Check that assets/hand_landmarker.task exists."
    )

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(options)


# -------------------- STATE VARIABLES --------------------

active = False

candidate_gesture = "UNKNOWN"
stable_count = 0

# Prevents a held gesture from triggering repeatedly.
last_triggered_gesture = None

# Spidey must be recognized before another navigation command
# can trigger after a previous navigation command.
command_armed = True


# -------------------- CAMERA --------------------

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    landmarker.close()
    raise RuntimeError(
        "Could not open camera. Check the camera index or camera access."
    )

print("\nGestureDeck started!")
print("Open palm = ACTIVATE")
print("Index finger = NEXT")
print("Index + middle fingers = PREVIOUS")
print("Spidey pose = READY FOR NEXT COMMAND")
print("Fist = DEACTIVATE")
print("Press Q in the camera window to quit.\n")


try:
    while True:
        success, frame = cap.read()

        if not success:
            print("Could not read a camera frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(mp_image)

        gesture = "UNKNOWN"
        thumb_ratio = 0.0

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            # Draw landmarks.
            for point in hand:
                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            # Draw connections.
            for start, end in HAND_CONNECTIONS:
                x1 = int(hand[start].x * frame.shape[1])
                y1 = int(hand[start].y * frame.shape[0])
                x2 = int(hand[end].x * frame.shape[1])
                y2 = int(hand[end].y * frame.shape[0])

                cv2.line(
                    frame, (x1, y1), (x2, y2),
                    (255, 0, 0), 2
                )

            finger_states, thumb_ratio = get_finger_states(hand)
            gesture = recognize_gesture(finger_states)

            cv2.putText(
                frame, f"Fingers: {finger_states}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

            cv2.putText(
                frame, f"Thumb ratio: {thumb_ratio:.2f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

        # -------------------- STABILITY CHECK --------------------

        if gesture == candidate_gesture:
            stable_count += 1
        else:
            candidate_gesture = gesture
            stable_count = 1

        # -------------------- GESTURE TRIGGER LOGIC --------------------

        if stable_count >= STABLE_FRAMES:

            if gesture == "UNKNOWN":
                # Unknown poses do not re-arm commands.
                pass

            elif gesture == "SPIDEY":
                if last_triggered_gesture != "SPIDEY":
                    if active:
                        command_armed = True
                        print("READY FOR NEXT COMMAND")
                    last_triggered_gesture = "SPIDEY"

            elif gesture == "ACTIVATE":
                if last_triggered_gesture != "ACTIVATE":
                    if not active:
                        active = True
                        command_armed = True
                        print("SYSTEM ACTIVATED")

                    last_triggered_gesture = "ACTIVATE"

            elif gesture == "DEACTIVATE":
                if last_triggered_gesture != "DEACTIVATE":
                    if active:
                        active = False
                        command_armed = True
                        print("SYSTEM DEACTIVATED")

                    last_triggered_gesture = "DEACTIVATE"

            elif gesture == "NEXT":
                if last_triggered_gesture != "NEXT":
                    if active and command_armed:
                        print("COMMAND: NEXT")
                        pyautogui.press("right")  # NEXT
                        command_armed = False
                    elif not active:
                        print("INACTIVE - COMMAND IGNORED")

                    last_triggered_gesture = "NEXT"

            elif gesture == "PREVIOUS":
                if last_triggered_gesture != "PREVIOUS":
                    if active and command_armed:
                        print("COMMAND: PREVIOUS")
                        pyautogui.press("left")   # PREVIOUS
                        command_armed = False
                    elif not active:
                        print("INACTIVE - COMMAND IGNORED")

                    last_triggered_gesture = "PREVIOUS"

        # -------------------- DISPLAY STATUS --------------------

        status = "ACTIVE" if active else "INACTIVE"
        status_color = (0, 255, 0) if active else (0, 0, 255)

        cv2.putText(
            frame, f"Status: {status}",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, status_color, 2
        )

        cv2.putText(
            frame, f"Gesture: {gesture}",
            (10, 95), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )

        cv2.putText(
            frame,
            f"Command ready: {'YES' if command_armed else 'SPIDEY NEEDED'}",
            (10, 125), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 255), 2
        )

        cv2.imshow("GestureDeck - Hand Gesture Controller", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("\nGestureDeck stopped.")
