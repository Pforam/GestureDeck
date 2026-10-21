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
THUMB_THRESHOLD = 0.50  # Kept for debug display
SHORTCUT_COOLDOWN = 1.5
MODE_TOGGLE_COOLDOWN = 1.5  # Seconds between mode switches
CURSOR_SMOOTHING = 0.35  # Higher = faster, lower = smoother
CURSOR_MARGIN = 0.08    # Ignore the outer 8% of the camera frame

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

# -------------------- GEOMETRY HELPERS --------------------
def distance(point_a, point_b):
    return math.sqrt(
        (point_a.x - point_b.x) ** 2
        + (point_a.y - point_b.y) ** 2
    )

def is_finger_open(hand, tip, pip):
    # Best when palm faces the camera and fingers point upward.
    return hand[tip].y < hand[pip].y

def get_finger_states(hand):
    # Tuple order: Thumb, Index, Middle, Ring, Pinky
    palm_width = distance(hand[5], hand[17])
    thumb_ratio = distance(hand[4], hand[5]) / palm_width if palm_width > 0 else 0.0
    thumb_tip_gap = distance(hand[4], hand[5])
    thumb_joint_gap = distance(hand[3], hand[5])
    thumb_open = thumb_tip_gap > (thumb_joint_gap * 1.15)

    index_open = is_finger_open(hand, 8, 6)
    middle_open = is_finger_open(hand, 12, 10)
    ring_open = is_finger_open(hand, 16, 14)
    pinky_open = is_finger_open(hand, 20, 18)

    return (
        int(thumb_open), int(index_open), int(middle_open),
        int(ring_open), int(pinky_open)
    ), thumb_ratio

# -------------------- GESTURE RECOGNITION --------------------
def recognize_gesture(finger_states):
    # Tuple order: (Thumb, Index, Middle, Ring, Pinky)
    if finger_states == (1, 1, 1, 1, 1):
        return "ACTIVATE"
    elif finger_states == (0, 1, 0, 0, 0):
        return "NEXT"
    elif finger_states == (0, 1, 1, 0, 0):
        return "PREVIOUS"
    elif finger_states == (1, 1, 0, 0, 1):
        return "SPIDEY"
    elif finger_states == (0, 0, 0, 0, 0):
        return "FIST"
    elif finger_states == (1, 0, 0, 0, 0):
        return "START_BEGINNING"
    elif finger_states == (1, 1, 0, 0, 0):
        return "START_CURRENT"
    elif finger_states[1:] == (1, 1, 1, 0):
        return "BLACK_SCREEN"
    elif finger_states == (1, 0, 0, 0, 1):
        return "EXIT_SLIDESHOW"
    elif finger_states == (0, 0, 0, 0, 1):
        return "CURSOR_TOGGLE"
    return "UNKNOWN"

def clamp(value, low, high):
    return max(low, min(high, value))

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
cursor_mode = False  # Toggle with pinky-only gesture; M remains a fallback
candidate_gesture = "UNKNOWN"
stable_count = 0
last_triggered_gesture = None
command_armed = True
last_shortcut_time = 0.0
last_mode_toggle_time = 0.0
toggle_release_required = False
smoothed_x = None
smoothed_y = None

# -------------------- CAMERA INITIALIZATION --------------------
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    landmarker.close()
    raise RuntimeError("Could not open camera. Check camera index/access.")

screen_width, screen_height = pyautogui.size()

print("\nGestureDeck started!")
print("Open palm = ACTIVATE")
print("Index finger = NEXT | Index + middle = PREVIOUS")
print("Spidey pose = re-arm navigation | Fist = DEACTIVATE")
print("Thumbs up = F5 | Thumb + index = Shift+F5")
print("Three fingers = BLACK SCREEN (B) | Thumb + pinky = ESC")
print("Pinky-only gesture = toggle CURSOR MODE (M is also a fallback).")
print("In cursor mode, point with your index fingertip; other commands are paused.")
print("Press Q in the camera window to quit.\n")

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read a camera frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        gesture = "UNKNOWN"
        thumb_ratio = 0.0
        finger_states = (0, 0, 0, 0, 0)
        hand = None

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            for point in hand:
                px = int(point.x * frame.shape[1])
                py = int(point.y * frame.shape[0])
                cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)

            for start, end in HAND_CONNECTIONS:
                x1 = int(hand[start].x * frame.shape[1])
                y1 = int(hand[start].y * frame.shape[0])
                x2 = int(hand[end].x * frame.shape[1])
                y2 = int(hand[end].y * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            finger_states, thumb_ratio = get_finger_states(hand)
            gesture = recognize_gesture(finger_states)

            # Cursor movement is available only when cursor mode is toggled on.
            # Landmark 8 is the index fingertip. Frame is mirrored, so the
            # displayed x-coordinate and mouse movement both feel natural.
            if cursor_mode and hand is not None:
                margin = CURSOR_MARGIN
                norm_x = (hand[8].x - margin) / (1.0 - 2.0 * margin)
                norm_y = (hand[8].y - margin) / (1.0 - 2.0 * margin)
                target_x = clamp(norm_x, 0.0, 1.0) * (screen_width - 1)
                target_y = clamp(norm_y, 0.0, 1.0) * (screen_height - 1)

                if smoothed_x is None:
                    smoothed_x, smoothed_y = target_x, target_y
                else:
                    smoothed_x += CURSOR_SMOOTHING * (target_x - smoothed_x)
                    smoothed_y += CURSOR_SMOOTHING * (target_y - smoothed_y)

                pyautogui.moveTo(int(smoothed_x), int(smoothed_y), _pause=False)
                tip_x = int(hand[8].x * frame.shape[1])
                tip_y = int(hand[8].y * frame.shape[0])
                # Bold target marker in the camera preview (not the OS cursor itself).
                cv2.circle(frame, (tip_x, tip_y), 20, (0, 255, 0), 4)
                cv2.circle(frame, (tip_x, tip_y), 5, (0, 255, 255), -1)
                cv2.line(frame, (tip_x - 27, tip_y), (tip_x + 27, tip_y), (0, 255, 0), 3)
                cv2.line(frame, (tip_x, tip_y - 27), (tip_x, tip_y + 27), (0, 255, 0), 3)

            cv2.putText(
                frame, f"Fingers (T,I,M,R,P): {finger_states}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2
            )
            cv2.putText(
                frame, f"Thumb ratio: {thumb_ratio:.2f}",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2
            )

        # Reset the toggle lock only after the pinky gesture disappears
        # if gesture != "CURSOR_TOGGLE":
        #     toggle_release_required = False

        # Stable gesture detection runs in both modes so the pinky-only
        # gesture can turn cursor mode on AND off. Other commands are paused
        # while cursor mode is active.
        if gesture != "CURSOR_TOGGLE":
            toggle_release_required = False

        if gesture == candidate_gesture:
            stable_count += 1
        else:
            candidate_gesture = gesture
            stable_count = 1

        
        
        if stable_count >= STABLE_FRAMES:
            if gesture == "CURSOR_TOGGLE":
                now = time.monotonic()

                if (
                    not toggle_release_required
                    and now - last_mode_toggle_time >= MODE_TOGGLE_COOLDOWN
                    and candidate_gesture == "CURSOR_TOGGLE"
                ):
                    cursor_mode = not cursor_mode
                    last_mode_toggle_time = now
                    toggle_release_required = True
                    last_triggered_gesture = "CURSOR_TOGGLE"

                    smoothed_x = None
                    smoothed_y = None

                    if cursor_mode:
                        print("CURSOR MODE ON - point with index fingertip")
                    else:
                        print("GESTURE MODE ON - gesture commands resumed")

            
            elif not cursor_mode:
                if gesture == "UNKNOWN":
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
                    "EXIT_SLIDESHOW",
                }:
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

        status = "ACTIVE" if active else "INACTIVE"
        status_color = (0, 255, 0) if active else (0, 0, 255)
        cv2.putText(
            frame, f"Status: {status}",
            (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, status_color, 2
        )
        mode_text = "CURSOR MODE - pinky-only to return" if cursor_mode else "GESTURE MODE - pinky-only for cursor"
        cv2.putText(
            frame, mode_text, (10, 86), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 255), 2
        )
        cv2.putText(
            frame, f"Gesture: {gesture}", (10, 116),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        ready_text = "YES" if command_armed else "SPIDEY NEEDED"
        cv2.putText(
            frame, f"Navigation ready: {ready_text}", (10, 146),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
        )

        cv2.imshow("GestureDeck - Hand Gesture Controller", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("m"):
            cursor_mode = not cursor_mode
            candidate_gesture = "UNKNOWN"
            stable_count = 0
            last_triggered_gesture = None
            smoothed_x = None
            smoothed_y = None
            print("CURSOR MODE ON - point with index fingertip; commands paused." if cursor_mode
                  else "GESTURE MODE ON - gesture commands resumed.")
        elif key == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("\nGestureDeck stopped.")
