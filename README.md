# Traffic Clearance System

A smart traffic management system that detects emergency vehicles and automatically manages traffic signals to provide priority passage for emergency services.

## Overview

This project uses computer vision and deep learning to detect emergency vehicles (ambulances, fire trucks, police cars) in traffic videos and intelligently control traffic signals to ensure rapid clearance of emergency vehicles through intersections.

## Features

- **Emergency Vehicle Detection**: Detects emergency vehicles using YOLO object detection combined with color-based filtering (red and blue detection)
- **Real-time Signal Management**: Automatically switches traffic signals to green when emergency vehicles are detected
- **Audio Alert System**: Plays beep alerts when emergency vehicles are detected
- **Video Processing**: Processes video files to generate annotated outputs
- **Multi-model Support**: Works with both YOLOv3 and YOLOv8 models

## Requirements

- Python 3.8+
- ultralytics (YOLO implementation)
- opencv-python (cv2)
- numpy
- supervision
- requests
- torch (for GPU acceleration, optional)

## Installation

1. **Clone or setup the project**
   ```bash
   cd "Traffic clearance"
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv yolov8-env
   yolov8-env\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained models**
   - YOLOv8 nano model: `yolov8n.pt` (automatically downloaded on first run)
   - YOLOv3 model: `yolov3.pt` (available in project root)

## Usage

### Basic Usage

```bash
python main.py
```

The script will:
1. Load the YOLOv8 nano model
2. Process video input (webcam or video file)
3. Detect vehicles and identify emergency vehicles
4. Control traffic signals accordingly
5. Save annotated video to the `runs/` directory

### Configuration

Modify the following in `main.py` to customize behavior:

- **Video Input**: Change the video source (webcam or file path)
- **Model**: Switch between YOLOv8 and YOLOv3 models
- **Detection Thresholds**: Adjust color detection thresholds in the `is_emergency_vehicle()` function
- **Signal Timing**: Modify signal state durations as needed

## Project Structure

```
Traffic clearance/
├── main.py                 # Main script for traffic clearance detection
├── download.py             # Model download utility
├── requirements.txt        # Python dependencies
├── TODO.md                # Project tasks and progress
├── README.md              # This file
├── yolov8n.pt             # YOLOv8 nano pre-trained model
├── yolov3.pt              # YOLOv3 pre-trained model
├── yolov8-env/            # Virtual environment directory
└── runs/                  # Output directory for processed videos
    ├── detect/            # Object detection outputs
    └── track/             # Object tracking outputs
```

## How It Works

### Emergency Vehicle Detection

The system uses a two-stage detection approach:

1. **Object Detection**: YOLO detects all vehicles in the frame
2. **Color Filtering**: Crops the bounding box area and analyzes HSV color space to identify red and blue components typical of emergency vehicle lights

### Signal Management

When an emergency vehicle is detected:
- Traffic signal switches to green for that direction
- Audio beep alert is triggered
- Vehicle is tracked across frames
- Detection is annotated on video output

## Models Used

- **YOLOv8 Nano (yolov8n.pt)**: Lightweight, fast inference, suitable for real-time processing
- **YOLOv3 (yolov3.pt)**: More robust, higher accuracy, requires more computational resources

## Output

Processed videos are saved to the `runs/` directory:
- Detection frames: `runs/detect/predict/`
- Tracking frames: `runs/track/`

Each output includes:
- Bounding boxes around detected vehicles
- Labels identifying vehicle types
- Signal state annotations
- Tracking IDs for vehicles

## Performance Considerations

- **GPU Acceleration**: Install PyTorch with CUDA for faster processing
- **Model Selection**: YOLOv8 nano is faster; use full YOLOv8 or YOLOv3 for better accuracy
- **Input Resolution**: Lower resolution inputs process faster but may reduce detection accuracy

## Future Enhancements

- [ ] Real-time traffic flow optimization
- [ ] Multi-lane detection and management
- [ ] Integration with actual traffic signal systems
- [ ] Machine learning-based signal timing optimization
- [ ] Web dashboard for monitoring
- [ ] Support for additional emergency vehicle types
- [ ] Performance metrics and statistics tracking

## Troubleshooting

### Model Loading Issues
```bash
# Manually download models
python download.py
```

### GPU Not Running
- Ensure NVIDIA drivers are installed
- Install PyTorch with CUDA support:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

### Low FPS
- Try YOLOv8 nano model for faster inference
- Reduce input video resolution
- Enable GPU acceleration

## Author

Created for traffic management and emergency response optimization.

## Contact

For questions or improvements, please refer to the TODO.md file for current development status.
