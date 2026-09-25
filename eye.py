"""Hands-free mouse control using MediaPipe Face Mesh and a webcam.

Right iris: move cursor. Left eye blink: left click.
Right eye blink while looking up/down: scroll.
Press Esc in the preview window to quit.
"""

import time

import cv2
import mediapipe as mp
import pyautogui


# Tune these values for your face and webcam if needed.
BLINK_EAR_THRESHOLD = 0.19
GAZE_THRESHOLD = 0.012
ACTION_COOLDOWN = 0.8
SCROLL_AMOUNT = 3
CURSOR_SMOOTHING = 0.42
# Keep mapped positions away from the screen corners, where PyAutoGUI's
# fail-safe is intentionally triggered.
SAFE_EDGE_MARGIN = 12
# Camera-frame active region (normalized 0..1): left, right, top, bottom.
# Iris movement inside this box maps across the full screen; outside it clamps
# to the nearest edge. Adjust these bounds for your webcam framing.
ACTIVE_REGION = (0.48, 0.52, 0.35, 0.45)

# Face Mesh eye landmarks. The camera preview is mirrored for natural control.
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
RIGHT_IRIS = (474, 475, 476, 477)


def eye_aspect_ratio(landmarks, eye, width, height):
    """Return an eye openness ratio that is less sensitive to face distance."""
    points = [
        (landmarks[index].x * width, landmarks[index].y * height)
        for index in eye
    ]
    vertical_a = ((points[1][0] - points[5][0]) ** 2 +
                  (points[1][1] - points[5][1]) ** 2) ** 0.5
    vertical_b = ((points[2][0] - points[4][0]) ** 2 +
                  (points[2][1] - points[4][1]) ** 2) ** 0.5
    horizontal = ((points[0][0] - points[3][0]) ** 2 +
                  (points[0][1] - points[3][1]) ** 2) ** 0.5
    if horizontal == 0:
        return 1.0
    return (vertical_a + vertical_b) / (2.0 * horizontal)


def right_iris_position(landmarks):
    """Return the right iris center as normalized camera-frame x and y."""
    iris_x = sum(landmarks[index].x for index in RIGHT_IRIS) / len(RIGHT_IRIS)
    iris_y = sum(landmarks[index].y for index in RIGHT_IRIS) / len(RIGHT_IRIS)
    return iris_x, iris_y


def map_active_region(value, region_min, region_max, screen_size):
    """Map a camera coordinate to screen pixels, away from fail-safe corners."""
    fraction = (value - region_min) / (region_max - region_min)
    fraction = max(0.0, min(1.0, fraction))
    margin = min(SAFE_EDGE_MARGIN, max(0, (screen_size - 1) // 2))
    return int(margin + fraction * (screen_size - 1 - 2 * margin))


def main():
    screen_width, screen_height = pyautogui.size()
    # DirectShow is generally lower latency for webcams on Windows. Fall back
    # to OpenCV's default backend if DirectShow cannot open the device.
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam 0. Check that it is connected and not in use.")

    # Smaller frames reduce processing time; a short buffer avoids showing stale video.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    last_click = 0.0
    last_scroll = 0.0
    previous_left_closed = False
    previous_right_closed = False
    gaze_y = 0.0
    gaze_baseline = None
    calibration_total = 0.0
    calibration_frames = 0
    smoothed_x = None
    smoothed_y = None
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Webcam frame could not be read; stopping.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb_frame)

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                now = time.monotonic()

                # Move using the center of the right iris, with exponential smoothing.
                # Map iris camera coordinates through the configurable active region.
                iris_x, iris_frame_y = right_iris_position(landmarks)
                region_left, region_right, region_top, region_bottom = ACTIVE_REGION
                target_x = map_active_region(iris_x, region_left, region_right, screen_width)
                target_y = map_active_region(iris_frame_y, region_top, region_bottom, screen_height)
                if smoothed_x is None:
                    smoothed_x, smoothed_y = target_x, target_y
                else:
                    smoothed_x += CURSOR_SMOOTHING * (target_x - smoothed_x)
                    smoothed_y += CURSOR_SMOOTHING * (target_y - smoothed_y)
                pyautogui.moveTo(int(smoothed_x), int(smoothed_y))

                left_closed = eye_aspect_ratio(landmarks, LEFT_EYE, width, height) < BLINK_EAR_THRESHOLD
                right_closed = eye_aspect_ratio(landmarks, RIGHT_EYE, width, height) < BLINK_EAR_THRESHOLD
                if not right_closed:
                    # Keep the last usable gaze measurement: the iris can be
                    # hidden or displaced by eyelids while the eye is closed.
                    # Calibrate vertical gaze from iris position only.
                    gaze_y = iris_frame_y
                    if gaze_baseline is None:
                        # Start while looking straight ahead; learn this user's
                        # natural iris position to make up/down gestures relative.
                        calibration_total += gaze_y
                        calibration_frames += 1
                        if calibration_frames >= 25:
                            gaze_baseline = calibration_total / calibration_frames

                # Trigger once on the open-to-closed edge, then wait for reopening.
                if left_closed and not previous_left_closed and now - last_click >= ACTION_COOLDOWN:
                    pyautogui.click()
                    last_click = now
                    cv2.putText(frame, "CLICK", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 255), 2)

                if (gaze_baseline is not None and right_closed and not previous_right_closed
                        and now - last_scroll >= ACTION_COOLDOWN):
                    relative_gaze = gaze_y - gaze_baseline
                    if relative_gaze < -GAZE_THRESHOLD:
                        pyautogui.scroll(SCROLL_AMOUNT)
                        last_scroll = now
                        action = "SCROLL UP"
                    elif relative_gaze > GAZE_THRESHOLD:
                        pyautogui.scroll(-SCROLL_AMOUNT)
                        last_scroll = now
                        action = "SCROLL DOWN"
                    else:
                        action = "LOOK UP OR DOWN, THEN BLINK RIGHT"
                    cv2.putText(frame, action, (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2)

                if gaze_baseline is not None:
                    gaze_delta = gaze_y - gaze_baseline
                    gaze_label = "UP" if gaze_delta < -GAZE_THRESHOLD else (
                        "DOWN" if gaze_delta > GAZE_THRESHOLD else "CENTER")
                    blink_label = "CLOSED" if right_closed else "OPEN"
                    cv2.putText(frame, f"Right eye: {blink_label} | Gaze: {gaze_label}",
                                (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 2)

                previous_left_closed = left_closed
                previous_right_closed = right_closed

                if gaze_baseline is None:
                    cv2.putText(frame, "Look straight ahead to calibrate...",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 255, 255), 2)

                # Draw the iris and eye points used for tracking.
                for index in RIGHT_IRIS + LEFT_EYE + RIGHT_EYE:
                    point = landmarks[index]
                    cv2.circle(frame, (int(point.x * width), int(point.y * height)),
                               2, (0, 255, 255), -1)

            cv2.imshow("Eye controlled mouse - Esc to quit", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        camera.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
