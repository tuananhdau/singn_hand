import os
import cv2
import json
import numpy as np
import tensorflow as tf

from collections import deque

from ai.hand_detector import HandDetector
from ai.feature_extractor import extract_landmarks


# ==================================================
# ĐƯỜNG DẪN PROJECT
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "sign_model.keras"
)

LABELS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "labels.json"
)


# ==================================================
# CAMERA
# ==================================================

class Camera:

    def __init__(self):

        print("====================================")
        print(" SIGNBRIDGE AI - CAMERA")
        print("====================================")

        # ------------------------------------------
        # Kiểm tra model
        # ------------------------------------------

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Không tìm thấy classification model:\n"
                f"{MODEL_PATH}"
            )

        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(
                f"Không tìm thấy labels:\n"
                f"{LABELS_PATH}"
            )

        # ------------------------------------------
        # Load classification model
        # ------------------------------------------

        print("Đang load classification model...")

        self.model = tf.keras.models.load_model(
            MODEL_PATH
        )

        # ------------------------------------------
        # Load labels
        # ------------------------------------------

        with open(
            LABELS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            self.labels = json.load(file)

        print("Labels:", self.labels)

        # ------------------------------------------
        # Hand Landmarker
        # ------------------------------------------

        print("Đang load Hand Landmarker...")

        self.hand_detector = HandDetector(
            num_hands=1
        )

        # ------------------------------------------
        # Camera
        # ------------------------------------------

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():

            raise RuntimeError(
                "Không thể mở camera laptop."
            )

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )

        # ------------------------------------------
        # Prediction smoothing
        # ------------------------------------------

        self.prediction_history = deque(
            maxlen=7
        )

        self.current_label = "UNKNOWN"
        self.current_confidence = 0.0

        print("Camera đã sẵn sàng.")


    # ==================================================
    # PREDICT
    # ==================================================

    def predict(self, features):

        if features is None:
            return "UNKNOWN", 0.0

        try:

            X = np.array(
                features,
                dtype=np.float32
            ).reshape(1, -1)

            probabilities = self.model.predict(
                X,
                verbose=0
            )[0]

            index = int(
                np.argmax(probabilities)
            )

            confidence = float(
                probabilities[index]
            )

            label = self.labels[index]

            # --------------------------------------
            # Confidence threshold
            # --------------------------------------

            if confidence < 0.80:

                return "UNKNOWN", confidence

            return label, confidence

        except Exception as e:

            print(
                "Prediction error:",
                e
            )

            return "UNKNOWN", 0.0


    # ==================================================
    # SMOOTH PREDICTION
    # ==================================================

    def smooth_prediction(
        self,
        label,
        confidence
    ):

        if label == "UNKNOWN":

            return "UNKNOWN", confidence

        self.prediction_history.append(
            label
        )

        # Đếm số lần xuất hiện
        counts = {}

        for item in self.prediction_history:

            counts[item] = (
                counts.get(item, 0) + 1
            )

        # Label xuất hiện nhiều nhất
        best_label = max(
            counts,
            key=counts.get
        )

        return best_label, confidence


    # ==================================================
    # DRAW UI
    # ==================================================

    def draw_prediction(
        self,
        frame,
        label,
        confidence
    ):

        height, width, _ = frame.shape

        # ------------------------------------------
        # Header
        # ------------------------------------------

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 75),
            (20, 25, 35),
            -1
        )

        cv2.putText(
            frame,
            "SIGNBRIDGE AI",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        # ------------------------------------------
        # Prediction box
        # ------------------------------------------

        height, width, _ = frame.shape
        box_width = 300
        box_height = 130

        x1 = 25
        y1 = height - box_height - 25

        x2 = x1 + box_width
        y2 = y1 + box_height

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (20, 25, 35),
            -1
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        # ------------------------------------------
        # Prediction text
        # ------------------------------------------

        cv2.putText(
            frame,
            "Prediction",
            (x1 + 20, y1 + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 20, y1 + 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (255, 255, 255),
            3
        )

        # ------------------------------------------
        # Confidence
        # ------------------------------------------

        confidence_text = (
            f"{confidence * 100:.1f}%"
        )

        cv2.putText(
            frame,
            confidence_text,
            (x1 + 145, y1 + 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2
        )

        return frame


    # ==================================================
    # GET FRAME
    # ==================================================

    def get_frame(self):

        success, frame = self.camera.read()

        if not success:

            return None

        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )

        # ------------------------------------------
        # MediaPipe Hand Landmarker
        # ------------------------------------------

        results = self.hand_detector.process(
            frame
        )

        # ------------------------------------------
        # Draw landmarks
        # ------------------------------------------

        frame = self.hand_detector.draw_landmarks(
            frame,
            results
        )

        # ------------------------------------------
        # Extract features
        # ------------------------------------------

        features = extract_landmarks(
            results
        )

        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        if features is not None:

            label, confidence = self.predict(
                features
            )

            label, confidence = self.smooth_prediction(
                label,
                confidence
            )

            self.current_label = label
            self.current_confidence = confidence

        else:

            self.current_label = "NO HAND"
            self.current_confidence = 0.0

            self.prediction_history.clear()

        # ------------------------------------------
        # Draw prediction
        # ------------------------------------------

        frame = self.draw_prediction(
            frame,
            self.current_label,
            self.current_confidence
        )

        # ------------------------------------------
        # JPEG
        # ------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not success:

            return None

        return buffer.tobytes()


    # ==================================================
    # RELEASE
    # ==================================================

    def release(self):

        if self.camera.isOpened():

            self.camera.release()

        self.hand_detector.close()


# ==================================================
# CAMERA INSTANCE
# ==================================================

camera = Camera()


# ==================================================
# GENERATE FRAMES
# ==================================================

def generate_frames():

    while True:

        frame = camera.get_frame()

        if frame is None:

            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )