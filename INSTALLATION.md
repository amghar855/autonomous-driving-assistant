# Installation Guide

Follow these steps to set up and run the VisionGuard Traffic Intelligence project on your local machine.

## Prerequisites

Ensure you have the following installed:
- [Python 3.8+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)

## 1. Clone the Repository

Open your terminal or command prompt and run the following command to download the project files:

```bash
git clone https://github.com/yourusername/visionguard.git
cd visionguard
```

## 2. Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies and avoid conflicts with other projects.

### Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the beginning of your terminal prompt, indicating the environment is active.

## 3. Install Dependencies

Install the required Python libraries using `pip`:

```bash
pip install -r requirements.txt
```

This will install Flask, OpenCV, Ultralytics, and other necessary packages.

## 4. Run the Application

Start the Flask development server:

```bash
python app.py
```

## 5. Access the Dashboard

Once the server is running, open your web browser and go to:

`http://127.0.0.1:5000`

## Troubleshooting

- **"python is not recognized"**: Ensure Python is added to your system PATH during installation.
- **"ModuleNotFoundError"**: Make sure your virtual environment is activated and you have run `pip install -r requirements.txt`.
- **"Port already in use"**: If port 5000 is occupied, you can change the port in `app.py` or stop the other process using that port.
