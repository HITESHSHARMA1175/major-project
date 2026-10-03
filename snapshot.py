"""
Webcam Snapshot - Sign Language Detection
==========================================
Takes ONE photo from your webcam, runs MediaPipe + detection on it,
saves it as 'snapshot.jpg' and opens it automatically.
"""

import cv2
import numpy as np
import os
import sys
import traceback
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, 'action.h5')
SNAPSHOT_PATH = os.path.join(BASE_DIR, 'snapshot.jpg')
ACTIONS      = np.array(['hello', 'thanks', 'iloveyou'])
SEQUENCE_LEN = 30
COLORS       = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

mp_holistic  = mp.solutions.holistic
mp_drawing   = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    try:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    except Exception:
        pass
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10),  thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76),  thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


def extract_keypoints(results):
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)
    lh   = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    rh   = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


def prob_viz(res, actions, input_frame, colors):
    out = input_frame.copy()
    for i, prob in enumerate(res):
        cv2.rectangle(out, (0, 60 + i*40), (int(prob*100), 90 + i*40), colors[i], -1)
        cv2.putText(out, actions[i], (0, 85 + i*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    # ── Load model ────────────────────────────────────────────────────────────
    print("[*] Loading model...")
    model = Sequential([
        Input(shape=(SEQUENCE_LEN, 1662)),
        LSTM(64,  return_sequences=True,  activation='relu'),
        LSTM(128, return_sequences=True,  activation='relu'),
        LSTM(64,  return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(ACTIONS), activation='softmax'),
    ])
    model.load_weights(MODEL_PATH)
    print("[OK] Model loaded!")

    # ── Open webcam and warm it up ────────────────────────────────────────────
    print("[*] Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit("[ERROR] Cannot open webcam!")

    # Warm up camera (first few frames are dark/blurry)
    print("[*] Warming up camera (3 seconds)... get ready!")
    for _ in range(30):
        cap.read()

    # Show countdown window
    WIN = "Snapshot | Get ready!"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, 600)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)

    import time
    for count in [3, 2, 1]:
        ret, frame = cap.read()
        if ret:
            display = frame.copy()
            cv2.putText(display, str(count), (280, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 15, cv2.LINE_AA)
            cv2.putText(display, "Get ready!", (150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.imshow(WIN, display)
            cv2.waitKey(1000)

    # ── Take ONE photo ────────────────────────────────────────────────────────
    ret, frame = cap.read()
    cap.release()

    if not ret:
        sys.exit("[ERROR] Failed to capture photo!")

    print("[OK] Photo captured!")

    # ── Run MediaPipe on the single frame ─────────────────────────────────────
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        image, results = mediapipe_detection(frame, holistic)

    draw_styled_landmarks(image, results)

    # ── Run model prediction ──────────────────────────────────────────────────
    keypoints = extract_keypoints(results)
    # Repeat single frame to fill sequence (for snapshot demo)
    sequence  = np.array([keypoints] * SEQUENCE_LEN)
    res       = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
    predicted = ACTIONS[np.argmax(res)]
    confidence = float(res[np.argmax(res)])

    print(f"\n  Prediction: {predicted.upper()} ({confidence*100:.1f}% confidence)")

    # ── Overlay probability bars ──────────────────────────────────────────────
    image = prob_viz(res, ACTIONS, image, COLORS)

    # Top banner
    cv2.rectangle(image, (0, 0), (640, 50), (245, 117, 16), -1)
    cv2.putText(image, f"Detected: {predicted.upper()} ({confidence*100:.0f}%)",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Save snapshot ─────────────────────────────────────────────────────────
    cv2.imwrite(SNAPSHOT_PATH, image)
    print(f"\n[OK] Snapshot saved to: {SNAPSHOT_PATH}")

    # ── Show result window ────────────────────────────────────────────────────
    cv2.setWindowTitle(WIN, f"Result: {predicted.upper()} | Press any key to close")
    cv2.imshow(WIN, image)
    print("[*] Press any key in the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Open the saved file in Windows default viewer too
    os.startfile(SNAPSHOT_PATH)
    print("[OK] Done!")


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        input("\nPress Enter to close...")
