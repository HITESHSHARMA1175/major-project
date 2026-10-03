"""
Real-Time Sign Language Action Detection - YOUR webcam
=======================================================
Detects: hello | thanks | iloveyou

Controls (click the webcam window first):
  S = Save screenshot to your Desktop
  Q or ESC = Quit
"""

import cv2
import numpy as np
import os
import sys
import traceback
import time
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# ── Suppress TF noise ─────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DESKTOP      = os.path.join(os.path.expanduser('~'), 'Desktop')
MODEL_PATH   = os.path.join(BASE_DIR, 'action.h5')
ACTIONS      = np.array(['hello', 'thanks', 'iloveyou'])
SEQUENCE_LEN = 30
THRESHOLD    = 0.5
COLORS       = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
WIN_NAME     = 'Sign Language Detection  |  S=Save to Desktop  Q=Quit'

mp_holistic  = mp.solutions.holistic
mp_drawing   = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


# ── Helpers ───────────────────────────────────────────────────────────────────

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
            image, results.face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,
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
    pose = np.array([[r.x, r.y, r.z, r.visibility]
                     for r in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[r.x, r.y, r.z]
                     for r in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)
    lh   = np.array([[r.x, r.y, r.z]
                     for r in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    rh   = np.array([[r.x, r.y, r.z]
                     for r in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


def prob_viz(res, actions, input_frame, colors):
    out = input_frame.copy()
    for i, prob in enumerate(res):
        cv2.rectangle(out, (0, 60 + i*40), (int(prob*100), 90 + i*40), colors[i], -1)
        cv2.putText(out, actions[i], (0, 85 + i*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load model
    print(f"\n[*] Loading model: {MODEL_PATH}")
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
    print("[OK] Model loaded!\n")

    # Open webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit("[ERROR] Cannot open webcam!")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create window and force it visible
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 800, 600)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_TOPMOST, 1)

    print("[OK] Webcam open!")
    print(f"[*] Preview saved every second to: {PREVIEW_PATH}")
    print("[*] Press Q or ESC in the webcam window to quit\n")

    sequence     = []
    sentence     = []
    predictions  = []
    last_save    = time.time()

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Camera read failed!")
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LEN:]

            if len(sequence) == SEQUENCE_LEN:
                res      = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                pred_idx = int(np.argmax(res))
                predictions.append(pred_idx)

                if len(predictions) >= 10:
                    if np.unique(predictions[-10:])[0] == pred_idx:
                        if res[pred_idx] > THRESHOLD:
                            if not sentence or ACTIONS[pred_idx] != sentence[-1]:
                                sentence.append(ACTIONS[pred_idx])
                                print(f"  Detected: {ACTIONS[pred_idx]}")

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                image = prob_viz(res, ACTIONS, image, COLORS)

            # Top banner
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            # Status
            ready  = len(sequence) >= SEQUENCE_LEN
            status = "DETECTING" if ready else f"Buffering {len(sequence)}/{SEQUENCE_LEN}"
            color  = (0, 255, 80) if ready else (180, 180, 180)
            cv2.putText(image, status, (420, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

            # Show window
            cv2.imshow(WIN_NAME, image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                print("\n[OK] Quitting.")
                break
            elif key == ord('s'):
                # Auto-number: Desktop/sign_001.jpg, sign_002.jpg ...
                i = 1
                while True:
                    path = os.path.join(DESKTOP, f'sign_{i:03d}.jpg')
                    if not os.path.exists(path):
                        break
                    i += 1
                cv2.imwrite(path, image)
                print(f"\n[OK] Screenshot saved to DESKTOP: {path}")
                os.startfile(path)

    cap.release()
    cv2.destroyAllWindows()
    print("[OK] Done.")


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        input("\nPress Enter to close...")
