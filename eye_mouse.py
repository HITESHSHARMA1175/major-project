import cv2
import os
import sys
import time
import urllib.request
import pyautogui

# Disable PyAutoGUI fail-safe pause if desired, or handle smooth movement
pyautogui.FAILSAFE = False

# Try camera indices 0, 1, 2
def get_camera():
    for index in (0, 1, 2):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[INFO] Successfully opened camera at index {index}")
                return cap
            cap.release()
    raise RuntimeError("No working webcam found. Please check your camera connection.")

MODEL_FILE = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

def ensure_model_file():
    if not os.path.exists(MODEL_FILE):
        print(f"[INFO] Downloading MediaPipe Face Landmarker model to {MODEL_FILE}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("[INFO] Model downloaded successfully.")

def main():
    print("[INFO] Starting Eye-Guided Mouse Control System...")
    cam = get_camera()
    screen_w, screen_h = pyautogui.size()

    import mediapipe as mp
    use_tasks_api = False

    # Check for MediaPipe Tasks API vs Legacy Solutions API
    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        ensure_model_file()
        base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        use_tasks_api = True
        print("[INFO] Using MediaPipe Tasks API (FaceLandmarker)")
    except Exception as e:
        print(f"[INFO] Tasks API not available ({e}), trying legacy mp.solutions...")
        try:
            face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
            print("[INFO] Using MediaPipe Legacy Solutions API (FaceMesh)")
        except Exception as e2:
            raise RuntimeError(f"Failed to initialize MediaPipe Face Mesh: {e2}")

    print("[INFO] Press 'q' or ESC in the preview window to exit.")

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            print("[WARNING] Failed to capture frame from camera.")
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        landmarks = None
        if use_tasks_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)
            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
        else:
            output = face_mesh.process(rgb_frame)
            if output.multi_face_landmarks:
                landmarks = output.multi_face_landmarks[0].landmark

        if landmarks:
            # Iris tracking (landmarks 474 to 477)
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                if id == 1:
                    screen_x = screen_w * landmark.x
                    screen_y = screen_h * landmark.y
                    pyautogui.moveTo(screen_x, screen_y)

            # Eyelid blink detection (landmarks 145 and 159)
            left_eye = [landmarks[145], landmarks[159]]
            for landmark in left_eye:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

            # Blink click trigger
            if (left_eye[0].y - left_eye[1].y) < 0.004:
                print("[INFO] Blink detected! Triggering click.")
                pyautogui.click()
                pyautogui.sleep(1)

        cv2.imshow('Eye Controlled Mouse', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("[INFO] Exiting Eye-Guided Mouse Control.")
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
