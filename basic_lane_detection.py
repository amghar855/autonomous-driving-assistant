import cv2
import numpy as np
import sys
import os

def process_video(video_path, output_path='output.mp4'):
    """
    🔹 Function: process_video
    Description:
        Detects road lanes from a given video and saves the processed video.

    Parameters:
        video_path (str): Path to input video file (or 0 for webcam)
        output_path (str): Path to save processed output video

    Example:
        process_video("C:/path/video.mp4")
    """

    print("🔹 Python used:", sys.executable)

    if not os.path.exists(video_path) and not str(video_path).isdigit():
        print("❌ Error: Video file not found:", video_path)
        return

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Couldn't read video.")
        cap.release()
        return

    height, width = frame.shape[:2]

    # Define VideoWriter for output
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("✅ Processing started... Press 'q' to stop manually.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create mask for lane area
        mask = np.zeros((height, width), dtype=np.uint8)
        trapezoid = np.array([[(
            int(width * 0.1), height),
            (int(width * 0.9), height),
            (int(width * 0.6), int(height * 0.6)),
            (int(width * 0.4), int(height * 0.6))
        ]], dtype=np.int32)
        cv2.fillPoly(mask, trapezoid, 255)


        # Image preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (13, 13), 0)
        canny = cv2.Canny(blur, 50, 150)

        masked = cv2.bitwise_and(canny, mask)
        contours, _ = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        # Write to output video
        out.write(frame)

        # Display real-time preview
        cv2.imshow('Lane Detection', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("⏹️ Manually stopped by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Processing complete! Saved at: {output_path}")

process_video(r"C:\Users\Admin\Desktop\ofppt project\ofppt project\projet fin anne\trafic\Basic-Lane-Detection-main\solidYellowLeft.mp4",
              r'C:\Users\Admin\Desktop\ofppt project\ofppt project\projet fin anne\trafic\Basic-Lane-Detection-main')