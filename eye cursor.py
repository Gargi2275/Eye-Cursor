import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# --- Setup ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "face_landmarker.task" 
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

detector = FaceLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)
sw, sh = pyautogui.size()
pyautogui.PAUSE = 0 

# --- Parameters ---
smooth = 5
prev_x, prev_y = 0, 0
eye_baseline = None
is_closed = False
last_blink = 0
click_gap = 0.5 

# Corner sensitivity: lower values = less eye movement needed to reach edges
bounds = [0.3, 0.7] 

def get_screen_coords(x, y):
    """Maps camera coordinates to screen pixels."""
    tx = np.interp(x, bounds, [0, sw])
    ty = np.interp(y, bounds, [0, sh])
    return tx, ty

print("Running... Press ESC to stop.")

timestamp = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Process Frame
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
    
    timestamp += 1
    result = detector.detect_for_video(mp_img, timestamp)

    if result.face_landmarks:
        face = result.face_landmarks[0]
        
        # 1. Measure Eye Openness (Left Eye)
        top, bottom = face[159], face[145]
        eye_open_dist = abs(top.y - bottom.y)

        if eye_baseline is None: eye_baseline = eye_open_dist
        eye_baseline = eye_baseline * 0.99 + eye_open_dist * 0.01 
        
        blinking = eye_open_dist < (eye_baseline * 0.65)

        # 2. Movement (Freeze cursor if blinking to prevent 'slipping')
        if not blinking:
            iris = face[475]
            target_x, target_y = get_screen_coords(iris.x, iris.y)

            # Smooth movement logic
            curr_x = prev_x + (target_x - prev_x) / smooth
            curr_y = prev_y + (target_y - prev_y) / smooth
            
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

        # 3. Click Logic (Single vs Double)
        if blinking and not is_closed:
            is_closed = True
        elif not blinking and is_closed:
            # Eye just opened - check for double click timing
            now = time.time()
            if (now - last_blink) < click_gap:
                pyautogui.doubleClick()
                last_blink = 0 
            else:
                pyautogui.click()
                last_blink = now
            is_closed = False

        # Visual marker for debugging
        dot_color = (0, 0, 255) if blinking else (0, 255, 0)
        cv2.circle(frame, (int(face[475].x * w), int(face[475].y * h)), 4, dot_color, -1)

    cv2.imshow("Eye Track", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
