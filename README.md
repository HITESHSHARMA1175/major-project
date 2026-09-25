# 👁️ AI-Powered Eye-Controlled Mouse

A hands-free mouse control program built with Python, OpenCV, MediaPipe Face Mesh, and PyAutoGUI. It uses a webcam to move the cursor with the right iris, click with a left-eye blink, and scroll with a right-eye blink while looking up or down.

## Project Objective

This project explores accessible, camera-based computer interaction. Facial landmarks estimate iris position and eye openness; PyAutoGUI translates the detected gestures into mouse actions.

## Tech Stack

| Tool or library | Purpose |
| --- | --- |
| Python 3.12 | Runs the program |
| OpenCV | Captures and processes webcam frames |
| MediaPipe Face Mesh | Detects face, iris, and eyelid landmarks |
| PyAutoGUI | Moves the cursor, clicks, and scrolls |

## Features

| Feature | Description |
| --- | --- |
| Eye tracking | Tracks the right iris in the webcam image |
| Cursor control | Maps the iris position through a configurable camera active region to the screen |
| Blink detection | Uses eye aspect ratio (EAR) to detect eye closure |
| Left click | A left-eye blink triggers one click |
| Scroll | A right-eye blink scrolls up or down based on the iris position relative to a neutral-gaze calibration |
| Smoothing and cooldown | Smooths cursor movement and limits repeated click and scroll actions |
| Low-latency capture | Requests 640×480 video, a one-frame buffer, and DirectShow on Windows when available |

## Gesture Mapping

| Gesture | Action |
| --- | --- |
| Right iris movement | Move the cursor |
| Left-eye blink | Left click |
| Look up, then blink right eye | Scroll up |
| Look down, then blink right eye | Scroll down |

Scroll is sent to the window underneath the cursor. Move the cursor over the window you want to scroll.

## Installation

Python 3.12 is recommended. This script uses the legacy `mediapipe.solutions.face_mesh` API; MediaPipe 0.10.18 is a known compatible version for Python 3.12.

### Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install mediapipe==0.10.18 opencv-contrib-python pyautogui
```

### macOS or Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install mediapipe==0.10.18 opencv-contrib-python pyautogui
```

## Run

From the folder containing `eye.py`, run:

```bash
python eye.py
```

Look straight ahead during startup while the neutral-gaze baseline is collected. Press **Esc** in the preview window to quit.

## Configuration and Calibration

Edit the constants near the top of `eye.py` to adjust the controls:

- `ACTIVE_REGION`: normalized camera-frame bounds `(left, right, top, bottom)` mapped across the screen. Iris positions outside the region clamp to the closest screen edge.
- `CURSOR_SMOOTHING`: higher values make cursor motion follow the iris more quickly; lower values smooth it more.
- `BLINK_EAR_THRESHOLD`: eye aspect ratio below this value counts as a blink.
- `GAZE_THRESHOLD`: minimum iris movement from the startup neutral baseline required to classify an up or down look.
- `ACTION_COOLDOWN` and `SCROLL_AMOUNT`: set the repeat delay and scroll step size.

Use steady, even lighting and face the webcam directly. Glasses glare, camera angle, and individual eye shape may affect detection, so thresholds can need adjustment.

## Privacy

Webcam frames are processed locally and are not saved or uploaded by this program.

## License

This project is licensed under the MIT License.

## Author

Made by [HITESH SHARMA](https://github.com/HITESHSHARMA1175).
