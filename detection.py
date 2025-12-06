import os
import time
from typing import Generator, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class DetectionPipeline:
    """
    Aggregates YOLO vehicle/sign detection with lane detection overlays.
    Provides helpers for processing images, videos, and live camera feeds.
    """

    def __init__(
        self,
        car_model_path: str = os.path.join("models", "car_model.pt"),
        sign_model_path: str = os.path.join("models", "paneaux_detect.pt"),
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.car_model = YOLO(car_model_path)
        self.sign_model = YOLO(sign_model_path)

    @staticmethod
    def _draw_lane_overlay(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        trapezoid = np.array(
            [
                [
                    (int(width * 0.1), height),
                    (int(width * 0.9), height),
                    (int(width * 0.6), int(height * 0.6)),
                    (int(width * 0.4), int(height * 0.6)),
                ]
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, trapezoid, 255)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        masked_edges = cv2.bitwise_and(edges, mask)

        line_image = np.zeros_like(frame)
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=150)
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        combined = cv2.addWeighted(frame, 0.85, line_image, 1.0, 0)
        return combined

    @staticmethod
    def _annotate_with_stats(frame: np.ndarray, fps: float, inference_time: float) -> None:
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Inference: {inference_time * 1000:.1f} ms",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _run_yolo(self, model: YOLO, frame: np.ndarray, color: Tuple[int, int, int]) -> None:
        results = model.predict(frame, device=self.device, verbose=False)
        if not results:
            return

        result = results[0]
        names = result.names
        for box in result.boxes:
            coords = box.xyxy.cpu().numpy().astype(int)[0]
            x1, y1, x2, y2 = coords
            confidence = float(box.conf.cpu().numpy())
            cls_idx = int(box.cls.cpu().numpy())
            label = names.get(cls_idx, f"cls_{cls_idx}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label}: {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        start_time = time.time()

        frame_with_lanes = self._draw_lane_overlay(frame)
        self._run_yolo(self.car_model, frame_with_lanes, (0, 191, 255))
        self._run_yolo(self.sign_model, frame_with_lanes, (255, 0, 0))

        end_time = time.time()
        inference_time = end_time - start_time
        fps = 1.0 / max(inference_time, 1e-6)

        self._annotate_with_stats(frame_with_lanes, fps, inference_time)
        return frame_with_lanes, fps, inference_time

    def process_image(self, input_path: str, output_path: str) -> dict:
        frame = cv2.imread(input_path)
        if frame is None:
            raise ValueError(f"Unable to read image at {input_path}")

        processed_frame, fps, inference_time = self.process_frame(frame)
        cv2.imwrite(output_path, processed_frame)
        return {"fps": fps, "inference_time": inference_time}

    def process_video(
        self,
        input_source: str,
        output_path: str,
        limit_seconds: Optional[int] = None,
    ) -> dict:
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video source {input_source}")

        fps_input = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps_input, (width, height))

        frame_count = 0
        total_fps = 0.0
        total_inference = 0.0
        start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, fps, inference_time = self.process_frame(frame)
            out.write(processed_frame)

            frame_count += 1
            total_fps += fps
            total_inference += inference_time

            if limit_seconds:
                elapsed = time.time() - start
                if elapsed >= limit_seconds:
                    break

        cap.release()
        out.release()

        total_time = time.time() - start

        return {
            "frames": frame_count,
            "average_fps": (total_fps / frame_count) if frame_count else 0,
            "average_inference_time": (total_inference / frame_count) if frame_count else 0,
            "elapsed_time": total_time,
        }

    def live_frame_generator(self) -> Generator[bytes, None, None]:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Unable to access webcam.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, _, _ = self.process_frame(frame)
                ret, buffer = cv2.imencode(".jpg", processed_frame)
                if not ret:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
        finally:
            cap.release()


pipeline = DetectionPipeline()


