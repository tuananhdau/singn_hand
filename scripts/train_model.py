import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sign_model.keras"
)

LABELS_PATH = os.path.join(
    MODEL_DIR,
    "labels.json"
)


# ==================================================
# 2. CẤU HÌNH
# ==================================================

TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 32


# ==================================================
# 3. ĐỌC DATASET
# ==================================================

def load_dataset():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy dataset:\n{DATASET_PATH}"
        )

    print("Đang đọc dataset:")
    print(DATASET_PATH)

    df = pd.read_csv(DATASET_PATH)

    if "label" not in df.columns:
        raise ValueError(
            "File CSV phải có cột 'label'."
        )

    if df.empty:
        raise ValueError(
            "Dataset đang rỗng."
        )

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    # Chuyển features sang dạng số
    X = X.astype(np.float32)

    print("\nThông tin dataset:")
    print(f"- Số mẫu: {len(X)}")
    print(f"- Số features: {X.shape[1]}")
    print(f"- Các label: {sorted(set(y))}")

    print("\nSố lượng mẫu mỗi label:")
    print(df["label"].value_counts().sort_index())

    return X, y


# ==================================================
# 4. XÂY DỰNG MODEL
# ==================================================

def build_model(input_size, number_of_classes):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(input_size,)
        ),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(
            number_of_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==================================================
# 5. TRAIN MODEL
# ==================================================

def main():

    print("=" * 50)
    print(" SIGNBRIDGE AI - TRAIN CLASSIFICATION MODEL")
    print("=" * 50)

    # Đọc dữ liệu
    X, y_text = load_dataset()

    # Mã hóa label:
    # A -> 0
    # B -> 1
    # ...
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    labels = list(label_encoder.classes_)

    print("\nMapping label:")
    for index, label in enumerate(labels):
        print(f"{label} -> {index}")

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nChia dữ liệu:")
    print(f"- Train: {len(X_train)} mẫu")
    print(f"- Test : {len(X_test)} mẫu")

    # Tạo model
    model = build_model(
        input_size=X_train.shape[1],
        number_of_classes=len(labels)
    )

    print("\nCấu trúc model:")
    model.summary()

    # Tạo thư mục models nếu chưa có
    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # Dừng sớm nếu validation không cải thiện
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    # Lưu model tốt nhất
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )

    print("\nBắt đầu train...\n")

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stopping,
            checkpoint
        ],
        verbose=1
    )

    # Đánh giá trên tập test
    print("\nĐánh giá model trên tập test:")

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(f"Test loss     : {test_loss:.4f}")
    print(f"Test accuracy : {test_accuracy * 100:.2f}%")

    # Dự đoán tập test
    predictions = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        predictions,
        axis=1
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=labels,
            zero_division=0
        )
    )

    print("Accuracy tính bằng sklearn:")
    print(
        f"{accuracy_score(y_test, y_pred) * 100:.2f}%"
    )

    # Lưu danh sách label để realtime prediction sử dụng
    with open(
        LABELS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            labels,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nĐã lưu model:")
    print(MODEL_PATH)

    print("\nĐã lưu danh sách label:")
    print(LABELS_PATH)

    print("\nHoàn thành train model.")


if __name__ == "__main__":
    main()