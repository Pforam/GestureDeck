# GestureDeck

**Touch-free presentation control using hand gestures.**

GestureDeck is a Python-based computer vision project that explores hands-free presentation interaction. It uses a webcam to capture hand movements, detects hand landmarks, interprets gestures, and maps recognized actions to computer controls.

Developed collaboratively by **Foram Patel and Stuti**.

## Features

- Live webcam input
- Hand landmark detection using MediaPipe
- Image and frame processing with OpenCV
- Gesture-based computer interaction
- Desktop keyboard/mouse automation using PyAutoGUI

> The exact gesture mappings and available controls depend on the implementation in `src/hand_tracker.py`. Refer to the source code for the currently supported gestures.

## Technology

- Python
- OpenCV
- MediaPipe
- PyAutoGUI
- NumPy

## Project structure

```text
GestureDeck/
├── assets/
│   └── hand_landmarker.task
├── docs/
├── papers/
├── screenshots/
├── src/
│   ├── camera_test.py
│   └── hand_tracker.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/GestureDeck.git
cd GestureDeck
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Check your webcam

```bash
python src/camera_test.py
```

### 5. Run GestureDeck

```bash
python src/hand_tracker.py
```

Make sure `assets/hand_landmarker.task` is present. If the application reports that the model file cannot be found, check the asset path configured in the source code.

## Requirements

- A webcam
- Windows, macOS, or Linux with a supported Python and MediaPipe combination
- A desktop environment where keyboard/mouse automation is permitted
- A presentation application for testing presentation controls

## Notes

- Camera performance and landmark detection can vary with lighting, camera quality, hand position, and occlusion.
- PyAutoGUI sends input to the active desktop session. Close or switch away from sensitive applications before testing.
- Do not run gesture automation while entering passwords or handling other sensitive information.
- The `hand_landmarker.task` model asset is required if referenced by the application. Keep it in the expected location.

## Contributors

- **Foram Patel**
- **Stuti**

## Project status

An academic collaborative project exploring computer vision, hand tracking, gesture interpretation, and desktop automation.
