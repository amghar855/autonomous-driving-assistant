# VisionGuard Traffic Intelligence

VisionGuard is an advanced AI-powered application designed for real-time traffic analysis. It combines state-of-the-art object detection (YOLOv8) with classic computer vision techniques to detect vehicles, traffic signs, and road lanes, providing a comprehensive solution for improved road safety and autonomous driving assistance.

## 🚀 Features

- **Real-time Object Detection**: Identifies vehicles and traffic signs using YOLOv8 models.
- **Lane Detection**: Visualizes road lanes using advanced image processing algorithms.
- **Multi-Source Input**: Supports image uploads, video files, and live webcam feeds.
- **User Dashboard**: Secure user accounts to manage and view past detection results.
- **Performance Metrics**: Displays real-time FPS and inference time for every processed frame.
- **Dark Mode UI**: A modern, responsive interface designed for optimal visibility.

## 📂 Project Structure

```
projet_final/
├── app.py                 # Main Flask application entry point
├── detection.py           # Core detection pipeline (YOLO + OpenCV)
├── basic_lane_detection.py # Standalone lane detection script
├── models/                # Pre-trained YOLO models
│   ├── car_model.pt
│   └── paneaux_detect.pt
├── static/                # Static assets (CSS, JS, uploads, outputs)
├── templates/             # HTML templates for the web interface
├── instance/              # SQLite database
└── requirements.txt       # Python dependencies
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/amghar855/autonomous-driving-assistant.git
    cd autonomous-driving-assistant
    ```

2.  **Create and activate a virtual environment:**
    *   **Windows:**
        ```powershell
        python -m venv venv
        .\venv\Scripts\Activate
        ```
    *   **macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ How to Run

1.  **Start the Flask application:**
    ```bash
    python app.py
    ```

2.  **Access the dashboard:**
    Open your browser and navigate to `http://127.0.0.1:5000`.

3.  **Register/Login:**
    Create a new account to access the detection features.

## 📋 Requirements

*   Python 3.8+
*   Flask
*   OpenCV
*   Ultralytics (YOLOv8)
*   NumPy
*   SQLAlchemy

## 👤 Author

Developed by amghar abdennour.
For inquiries, please contact amgharabdennour7@gmail.com .
