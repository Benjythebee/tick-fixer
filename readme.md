# Tick-Fix 🎯

A real-time facial tic detection system using computer vision and machine learning. This project uses MediaPipe's FaceMesh solution to track facial landmarks and detect involuntary facial movements (tics) through webcam monitoring.

I made this because I had a bad tick that I really wanted to get reminders to stop it.

## 🌟 Features

- **Real-time Face Tracking**: Uses MediaPipe FaceMesh to track 3D facial landmarks
- **Intelligent Tic Detection**: Analyzes mouth width patterns to identify tic movements
- **Baseline Adaptive Algorithm**: Uses rolling median baseline to adapt to individual facial features
- **Smart Filtering**: Requires sustained pattern (10 frames) to avoid false positives
- **Windows Notifications**: Desktop notifications when tics are detected (using win11toast)
- **Performance Optimized**: Processes every 2nd frame for better performance
- **Debug Mode**: Optional visualization of landmarks and detection metrics

## 🔬 How It Works

### Detection Algorithm

The system detects facial tics using a multi-step approach:

1. **Landmark Tracking**: Tracks key mouth landmarks (left, right, upper, lower)
2. **Normalization**: Calculates mouth width as a ratio of face width (scale-independent)
3. **Baseline Calculation**: Maintains a 30-frame rolling history and calculates median baseline
4. **Pattern Detection**: Triggers when mouth width exceeds 110% of baseline
5. **Persistence Verification**: Requires pattern to persist for 10 consecutive frames
6. **Cooldown**: Enforces 2-second cooldown between detections to avoid duplicates

### Key Landmarks Used

- **Mouth Left** (61) & **Mouth Right** (291): Horizontal mouth width
- **Mouth Upper** (13) & **Mouth Lower** (14): Vertical mouth opening
- **Left Cheek** (172) & **Right Cheek** (397): Reference points for face width normalization

## 📋 Requirements

- Python 3.12+
- Webcam (configured for camera index 1 by default)
- Windows OS (for notifications - can be adapted for other platforms)

### Dependencies

```
mediapipe==0.10.14
opencv-contrib-python==4.12.0.88
numpy==2.2.6
win11toast
```

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Benjythebee/tick-fix.git
   cd tick-fix
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install mediapipe==0.10.14 opencv-contrib-python==4.12.0.88 numpy==2.2.6 win11toast
   ```

## 💻 Usage

### Basic Usage

Run the main script:
```bash
python main.py
```

Press **ESC** to exit.

### Configuration Options

Edit `main.py` to customize behavior:

```python
# Camera selection (default is 1, change to 0 for default webcam)
cap = cv2.VideoCapture(1)

# Performance settings
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame

# Detection parameters
HISTORY_SIZE = 30              # Frames for baseline calculation
HORIZONTAL_THRESHOLD = 1.10    # 10% increase threshold
TICK_DURATION_FRAMES = 10      # Minimum frames for valid detection
TICK_COOLDOWN = 2.0            # Seconds between detections

# Visualization options
DRAW_FULL_MESH = False  # Set True to see full face mesh
IS_DEBUG = True         # Set True to enable debug overlay and window
```

### Debug Mode

Enable debug mode to see:
- Live mouth width ratio
- Face width measurements
- Baseline ratio
- Tick counter progress
- FPS display
- Landmark labels

Set `IS_DEBUG = True` in `main.py` to enable.


## 📊 Algorithm Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HISTORY_SIZE` | 30 | Number of frames for baseline (≈1 second at 30fps) |
| `HORIZONTAL_THRESHOLD` | 1.10 | Multiplier for tic detection (10% increase) |
| `TICK_DURATION_FRAMES` | 10 | Frames pattern must persist |
| `TICK_COOLDOWN` | 2.0s | Minimum time between detections |

## 🛠️ Troubleshooting

### Camera Issues
- **Wrong camera**: Change `cv2.VideoCapture(1)` to `cv2.VideoCapture(0)` or other index
- **No camera found**: Ensure webcam is connected and not in use by other applications

### Performance Issues
- Increase `PROCESS_EVERY_N_FRAMES` to 3 or 4
- Set `DRAW_FULL_MESH = False`
- Set `IS_DEBUG = False` when not needed

### Detection Sensitivity
- **Too sensitive**: Increase `HORIZONTAL_THRESHOLD` (e.g., 1.15 for 15% increase)
- **Not sensitive enough**: Decrease `HORIZONTAL_THRESHOLD` (e.g., 1.05 for 5% increase)
- **False positives**: Increase `TICK_DURATION_FRAMES`

## 📝 Project Structure

```
tick-fix/
├── main.py              # Main application with tic detection
├── main2.py             # Placeholder for experiments
├── readme.md            # This file
├── .venv/               # Python virtual environment
└── .github/
    └── copilot-instructions.md  # Development guidelines
```

## 📄 License

This project is open source and available for personal and research use.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Cross-platform compatibility
- Additional tic detection patterns
- UI/UX enhancements
- Performance optimizations
- Documentation improvements

## 🙏 Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) by Google for facial landmark detection
- [OpenCV](https://opencv.org/) for computer vision tools
- Face mesh landmark visualization: [MediaPipe Face Mesh](https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)

---

**Note**: This project is intended for research and personal awareness. It is not a medical diagnostic tool. Consult healthcare professionals for medical concerns related to tics or movement disorders.