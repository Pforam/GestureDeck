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
THUMB_THRESHOLD = 0.50  # Kept for reference/debug display
SHORTCUT_COOLDOWN = 1.5  # Helps prevent duplicate shortcut presses

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
    # Works best when the palm faces the camera and fingers point upward.
    return hand[tip].y < hand[pip].y


def get_finger_states(hand):
    # Tuple order: Thumb, Index, Middle, Ring, Pinky
    palm_width = distance(hand[5], hand[17])

    if palm_width > 0:
        thumb_ratio = distance(hand[4], hand[5]) / palm_width
    else:
        thumb_ratio = 0.0

    # Compare the thumb tip's distance from the index-finger base with the
    # thumb IP joint's distance. This is less prone to false positives than
    # a single fixed thumb-distance threshold, though hand angle still matters.
    thumb_tip_gap = distance(hand[4], hand[5])
    thumb_joint_gap = distance(hand[3], hand[5])
    thumb_open = thumb_tip_gap > (thumb_joint_gap * 1.15)

    index_open = is_finger_open(hand, 8, 6)
    middle_open = is_finger_open(hand, 12, 10)
    ring_open = is_finger_open(hand, 16, 14)
    pinky_open = is_finger_open(hand, 20, 18)

    finger_states = (
        int(thumb_open),
        int(index_open),
        int(middle_open),
        int(ring_open),
        int(pinky_open),
    )
    return finger_states, thumb_ratio


# -------------------- GESTURE RECOGNITION --------------------

def recognize_gesture(finger_states):
    # Tuple order: (Thumb, Index, Middle, Ring, Pinky)
    if finger_states == (1, 1, 1, 1, 1):
        return "ACTIVATE"             # Open palm
    elif finger_states == (0, 1, 0, 0, 0):
        return "NEXT"                 # Index finger
    elif finger_states == (0, 1, 1, 0, 0):
        return "PREVIOUS"             # Index + middle
    elif finger_states == (1, 1, 0, 0, 1):
        return "SPIDEY"               # Existing re-arm gesture
    elif finger_states == (0, 0, 0, 0, 0):
        return "FIST"                 # Deactivate while active

    # New presentation controls
    elif finger_states == (1, 0, 0, 0, 0):
        return "START_BEGINNING"      # Thumbs up -> F5
    elif finger_states == (1, 1, 0, 0, 0):
        return "START_CURRENT"        # Thumb + index -> Shift+F5
    elif finger_states[1:] == (1, 1, 1, 0):
        return "BLACK_SCREEN"         # Index + middle + ring; ignore thumb
    elif finger_states == (1, 0, 0, 0, 1):
        return "EXIT_SLIDESHOW"       # Thumb + pinky -> Esc

    return "UNKNOWN"


# -------------------- MEDIAPIPE INITIALIZATION --------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Hand landmark model not found:\n{MODEL_PATH}\n"
        "Check that assets/hand_landmarker.task exists."
    )

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
)
landmarker = vision.HandLandmarker.create_from_options(options)


# -------------------- STATE VARIABLES --------------------

active = False
candidate_gesture = "UNKNOWN"
stable_count = 0
last_triggered_gesture = None
command_armed = True
last_shortcut_time = 0.0


# -------------------- CAMERA INITIALIZATION --------------------
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
print("Spidey pose = READY FOR NEXT NAVIGATION COMMAND")
print("Fist = DEACTIVATE (when active)")
print("Thumbs up = START FROM BEGINNING (F5)")
print("Thumb + index = START FROM CURRENT SLIDE (Shift+F5)")
print("Three fingers = BLACK SCREEN (B)")
print("Thumb + pinky = EXIT SLIDESHOW (Esc)")
print("Press Q in the camera window to quit.\n")



try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read a camera frame.")
            break

        # Mirror the camera image for a natural preview.
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        result = landmarker.detect(mp_image)

        gesture = "UNKNOWN"
        thumb_ratio = 0.0
        finger_states = (0, 0, 0, 0, 0)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            # Draw landmark points.
            for point in hand:
                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            # Draw hand connections.
            for start, end in HAND_CONNECTIONS:
                x1 = int(hand[start].x * frame.shape[1])
                y1 = int(hand[start].y * frame.shape[0])
                x2 = int(hand[end].x * frame.shape[1])
                y2 = int(hand[end].y * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            finger_states, thumb_ratio = get_finger_states(hand)
            gesture = recognize_gesture(finger_states)

            cv2.putText(
                frame, f"Fingers (T,I,M,R,P): {finger_states}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2,
            )
            cv2.putText(
                frame, f"Thumb ratio: {thumb_ratio:.2f}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2,
            )

        # -------------------- STABILITY CHECK --------------------

        if gesture == candidate_gesture:
            stable_count += 1
        else:
            candidate_gesture = gesture
            stable_count = 1

        # Process a gesture once it has remained stable for enough frames.
        if stable_count >= STABLE_FRAMES and gesture != last_triggered_gesture:
            if gesture == "UNKNOWN":
                # A stable neutral/unknown pose releases the one-shot lock.
                last_triggered_gesture = "UNKNOWN"

            elif gesture == "ACTIVATE":
                if not active:
                    active = True
                    command_armed = True
                    print("SYSTEM ACTIVATED")
                last_triggered_gesture = gesture

            elif gesture == "FIST":
                if active:
                    active = False
                    command_armed = True
                    print("SYSTEM DEACTIVATED")
                # When inactive, fist does not send Esc. Use thumb + pinky
                # for the dedicated slideshow-exit command instead.
                last_triggered_gesture = gesture

            elif gesture == "SPIDEY":
                if active:
                    command_armed = True
                    print("READY FOR NEXT NAVIGATION COMMAND")
                last_triggered_gesture = gesture

            elif gesture == "NEXT":
                if active and command_armed:
                    print("COMMAND: NEXT")
                    pyautogui.press("right")
                    command_armed = False
                elif not active:
                    print("INACTIVE - NAVIGATION COMMAND IGNORED")
                last_triggered_gesture = gesture

            elif gesture == "PREVIOUS":
                if active and command_armed:
                    print("COMMAND: PREVIOUS")
                    pyautogui.press("left")
                    command_armed = False
                elif not active:
                    print("INACTIVE - NAVIGATION COMMAND IGNORED")
                last_triggered_gesture = gesture

            elif gesture in {
                "START_BEGINNING", "START_CURRENT", "BLACK_SCREEN",
                "WHITE_SCREEN", "EXIT_SLIDESHOW",
            }:
                # Require activation and a short cooldown to guard against
                # brief landmark flicker causing the same shortcut twice.
                now = time.monotonic()
                if active:
                    if now - last_shortcut_time >= SHORTCUT_COOLDOWN:
                        if gesture == "START_BEGINNING":
                            print("COMMAND: START SLIDESHOW FROM BEGINNING (F5)")
                            pyautogui.press("f5")
                        elif gesture == "START_CURRENT":
                            print("COMMAND: START FROM CURRENT SLIDE (SHIFT+F5)")
                            pyautogui.hotkey("shift", "f5")
                        elif gesture == "BLACK_SCREEN":
                            print("COMMAND: BLACK SCREEN (B)")
                            pyautogui.press("b")
                        elif gesture == "EXIT_SLIDESHOW":
                            print("COMMAND: EXIT SLIDESHOW (ESC)")
                            pyautogui.press("esc")
                        last_shortcut_time = now
                    else:
                        print(f"COOLDOWN - {gesture} ignored to prevent duplicate press")
                else:
                    print(f"INACTIVE - {gesture} IGNORED; ACTIVATE FIRST")
                last_triggered_gesture = gesture

                # -------------------- DISPLAY STATUS --------------------
        status = "ACTIVE" if active else "INACTIVE"
        status_color = (0, 255, 0) if active else (0, 0, 255)
        cv2.putText(
            frame, f"Status: {status}",
            (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, status_color, 2,
        )
        cv2.putText(
            frame, f"Gesture: {gesture}",
            (10, 88), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2,
        )
        ready_text = "YES" if command_armed else "SPIDEY NEEDED"
        cv2.putText(
            frame, f"Navigation ready: {ready_text}",
            (10, 118), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 255), 2,
        )

        cv2.imshow("GestureDeck - Hand Gesture Controller", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("\nGestureDeck stopped.")




