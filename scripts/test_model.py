import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==================================================
# 1. ĐƯỜNG DẪN
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "landmarks.csv"
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
# 2. CẤU HÌNH
# ==================================================

TEST_SIZE = 0.2
RANDOM_STATE = 42


# ==================================================
# 3. LOAD DATASET
# ==================================================

def load_dataset():

    print("\n[1] Đọc dataset...")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy dataset:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    if "label" not in df.columns:
        raise ValueError(
            "Dataset phải có cột 'label'."
        )

    if df.empty:
        raise ValueError(
            "Dataset đang rỗng."
        )

    X = df.drop(
        columns=["label"]
    ).values.astype(np.float32)

    y_text = df["label"].values

    print(f"Dataset: {DATASET_PATH}")
    print(f"Số mẫu: {len(X)}")
    print(f"Số features: {X.shape[1]}")

    print("\nSố lượng mẫu từng label:")

    counts = df["label"].value_counts().sort_index()

    for label, count in counts.items():
        print(f"  {label}: {count}")

    return X, y_text


# ==================================================
# 4. LOAD LABELS
# ==================================================

def load_labels():

    if os.path.exists(LABELS_PATH):

        with open(
            LABELS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            labels = json.load(file)

        return labels

    return None


# ==================================================
# 5. MAIN
# ==================================================

def main():

    print("=" * 60)
    print(" SIGNBRIDGE AI - MODEL TEST")
    print("=" * 60)

    # ------------------------------------------------
    # Kiểm tra model
    # ------------------------------------------------

    print("\n[0] Kiểm tra model...")

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Không tìm thấy model:\n{MODEL_PATH}"
        )

    print(f"Model: {MODEL_PATH}")

    # ------------------------------------------------
    # Load dataset
    # ------------------------------------------------

    X, y_text = load_dataset()

    # ------------------------------------------------
    # Encode label
    # ------------------------------------------------

    encoder = LabelEncoder()

    y = encoder.fit_transform(
        y_text
    )

    labels = list(
        encoder.classes_
    )

    print("\nMapping label:")

    for index, label in enumerate(labels):
        print(
            f"  {label} -> {index}"
        )

    # ------------------------------------------------
    # Chia TEST giống train_model.py
    # ------------------------------------------------

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n[2] Tập TEST")
    print(
        f"Số mẫu test: {len(X_test)}"
    )

    # ------------------------------------------------
    # Load model
    # ------------------------------------------------

    print("\n[3] Load model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Load model thành công.")

    # ------------------------------------------------
    # Predict
    # ------------------------------------------------

    print("\n[4] Đang dự đoán...")

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    # ------------------------------------------------
    # Accuracy
    # ------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\n" + "=" * 60)
    print(" KẾT QUẢ")
    print("=" * 60)

    print(
        f"\nAccuracy: {accuracy * 100:.2f}%"
    )

    # ------------------------------------------------
    # Classification report
    # ------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=labels,
            zero_division=0
        )
    )

    # ------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("Confusion Matrix:")

    print(cm)

    # ------------------------------------------------
    # Hiển thị dạng dễ đọc
    # ------------------------------------------------

    print("\nChi tiết Confusion Matrix:")

    print(
        "      " +
        " ".join(
            f"{label:>5}"
            for label in labels
        )
    )

    for i, label in enumerate(labels):

        row = " ".join(
            f"{value:>5}"
            for value in cm[i]
        )

        print(
            f"{label:>5} {row}"
        )

    # ------------------------------------------------
    # Một số dự đoán mẫu
    # ------------------------------------------------

    print("\nMột số dự đoán:")

    sample_count = min(
        20,
        len(X_test)
    )

    for i in range(sample_count):

        actual = labels[y_test[i]]
        predicted = labels[y_pred[i]]

        confidence = float(
            probabilities[i][y_pred[i]]
        )

        status = "✓" if actual == predicted else "✗"

        print(
            f"{status} "
            f"Thực tế: {actual} | "
            f"Dự đoán: {predicted} | "
            f"Confidence: {confidence * 100:.2f}%"
        )

    print("\n" + "=" * 60)
    print(" TEST HOÀN TẤT")
    print("=" * 60)


if __name__ == "__main__":
    main()