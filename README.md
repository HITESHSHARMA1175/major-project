# 👁️ AI-Powered Eye-Controlled Mouse

A hands-free mouse control project built with Python, OpenCV, MediaPipe, and PyAutoGUI. It uses a webcam to track the right iris for cursor movement and a left-eye blink for clicking.

## 🧠 Project Objective

The project explores hands-free computer interaction through eye movement and blinks. The current program moves the cursor with the right iris and clicks with a left-eye blink. Right-eye blink scrolling is a planned enhancement and is not implemented in the current code.

## 🧰 Tech Stack

| Tool or library | Purpose |
| --- | --- |
| Python | Runs the application |
| OpenCV | Captures and processes webcam frames |
| MediaPipe | Detects face and eye landmarks |
| PyAutoGUI | Moves the cursor and sends mouse clicks |

## 🔍 Use Cases

| Use case | Description |
| --- | --- |
| Accessibility exploration | Experiments with hands-free cursor control |
| Multitasking | Moves the cursor while hands are occupied |
| Human-computer interaction | Demonstrates camera-based input |
| Computer vision projects | Provides a starting point for eye-landmark experiments |

## ⚙️ Current Features

| Feature | Description |
| --- | --- |
| 👁️ Eye tracking | Tracks iris landmarks from a live webcam feed |
| 🖱️ Cursor control | Maps right-iris position in the camera frame to the screen |
| 👁️ Blink click | Uses left-eye landmarks to trigger a left click |
| ⏱️ Click delay | Waits one second after a detected click to limit repeated clicks |
| 📷 Camera selection | Tries camera indexes 0, 1, and 2 |
| 🧠 Model setup | Downloads the MediaPipe face-landmark model if it is missing |

## 🔄 Eye Gesture Mapping

| Gesture | Action |
| --- | --- |
| Right iris movement | Move the mouse cursor |
| Left-eye blink | Left click |
| Right-eye blink plus looking up or down | Planned scrolling feature; not currently available |

## 💻 Architecture

```text
[ Webcam Feed ]
      ↓
[ OpenCV + MediaPipe Face Landmarker ]
      ↓
[ Iris and Eyelid Landmark Detection ]
      ↓
 ┌─────────────────────────┐
 │ Cursor movement         │ ← Right iris
 ├─────────────────────────┤
 │ Left click              │ ← Left-eye blink
 └─────────────────────────┘
      ↓
[ PyAutoGUI → Operating System ]
```

## 📆 Installation

### 1. Clone the repository

```bash
git clone https://github.com/wasimtikki120/Eye-Guided-Mouse-Control-The-Future-of-User-Interface.git
cd Eye-Guided-Mouse-Control-The-Future-of-User-Interface
```

### 2. Create a virtual environment and install dependencies

Python 3.12 is recommended.

#### Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install opencv-contrib-python mediapipe pyautogui
```

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install opencv-contrib-python mediapipe pyautogui
```

The first run downloads `face_landmarker.task` into the current working directory if it is not already there. An internet connection is needed for this initial download.

## ▶️ Run the Program

From the repository directory, run:

```bash
python eye_mouse.py
```

Alternatively, `python code.py` launches the same program. Press **Q** or **Esc** in the preview window to exit.

## ⚠️ Notes and Calibration Tips

- Use even lighting and face the webcam directly.
- Keep your face and eyes visible in the camera preview.
- Blink detection uses a fixed landmark threshold in `eye_mouse.py`; accuracy can vary by person and camera.
- Glasses glare, camera angle, and low light can affect tracking.
- PyAutoGUI's fail-safe is disabled by the current code. Use **Q** or **Esc** in the preview to stop the program.
- Webcam video is processed locally; the program does not save or upload the camera feed.

## 💡 Future Enhancements

- Scroll up and down using right-eye blinks and gaze direction
- Add a calibration wizard and configurable active screen region
- Improve blink detection and cursor smoothing
- Add a GUI overlay for click feedback
- Add voice feedback
- Package the application for Windows and macOS

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## 🙌 Author

Made by [HITESH SHARMA](https://github.com/HITESHSHARMA1175).
