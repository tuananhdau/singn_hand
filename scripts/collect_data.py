import cv2
import csv
import os
import time
import sys

# Cho phép Python tìm thư mục ai/ ở thư mục gốc project
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from ai.hand_detector import HandDetector
from ai.feature_extractor import extract_landmarks


# =========================================================
# CẤU HÌNH
# =========================================================

DATASET_DIR = "dataset"
CSV_FILE = os.path.join(DATASET_DIR, "landmarks.csv")

SAMPLES_PER_LABEL = 500

# Các nhãn cần thu thập
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# =========================================================
# TẠO THƯ MỤC DATASET
# =========================================================

os.makedirs(DATASET_DIR, exist_ok=True)


# =========================================================
# TẠO FILE CSV
# =========================================================

def create_csv():

    if os.path.exists(CSV_FILE):
        return

    header = ["label"]

    for i in range(21):
        header.extend([
            f"x{i}",
            f"y{i}",
            f"z{i}"
        ])

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)
        writer.writerow(header)


# =========================================================
# ĐẾM SỐ MẪU CỦA TỪNG LABEL
# =========================================================

def count_samples():

    counts = {
        label: 0
        for label in LABELS
    }

    if not os.path.exists(CSV_FILE):
        return counts

    with open(
        CSV_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            label = row["label"]

            if label in counts:
                counts[label] += 1

    return counts


# =========================================================
# LƯU MỘT SAMPLE
# =========================================================

def save_sample(label, features):

    with open(
        CSV_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [label] + features.tolist()
        )


# =========================================================
# HIỂN THỊ TEXT LÊN CAMERA
# =========================================================

def draw_text(
    frame,
    text,
    position,
    scale=0.7,
    thickness=2
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# =========================================================
# MAIN
# =========================================================

def main():

    create_csv()

    counts = count_samples()

    current_index = 0

    # Tìm label chưa hoàn thành đầu tiên
    for i, label in enumerate(LABELS):

        if counts[label] < SAMPLES_PER_LABEL:
            current_index = i
            break

    else:

        print("====================================")
        print("DATASET ĐÃ HOÀN THÀNH A-Z")
        print("====================================")
        return

    current_label = LABELS[current_index]

    print("====================================")
    print(" SIGNBRIDGE AI - DATASET COLLECTOR")
    print("====================================")
    print()
    print("Label hiện tại:", current_label)
    print("Số mẫu:", counts[current_label])
    print()
    print("SPACE : Lưu mẫu")
    print("N     : Label tiếp theo")
    print("R     : Reset label hiện tại")
    print("Q     : Thoát")
    print()

    # =====================================================
    # CAMERA
    # =====================================================

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("Không thể mở camera!")

        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    detector = HandDetector(
        num_hands=1
    )

    last_save_time = 0

    try:

        while True:

            success, frame = cap.read()

            if not success:

                print("Không thể đọc camera.")

                break

            # Lật camera giống webcam
            frame = cv2.flip(
                frame,
                1
            )

            # =============================================
            # MEDIAPIPE
            # =============================================

            results = detector.process(frame)

            # Vẽ landmark
            frame = detector.draw_landmarks(
                frame,
                results
            )

            # =============================================
            # EXTRACT FEATURES
            # =============================================

            features = extract_landmarks(
                results
            )

            # =============================================
            # THÔNG TIN TRÊN MÀN HÌNH
            # =============================================

            h, w, _ = frame.shape

            # Header
            cv2.rectangle(
                frame,
                (0, 0),
                (w, 115),
                (20, 20, 20),
                -1
            )

            draw_text(
                frame,
                "SIGNBRIDGE AI - DATASET COLLECTOR",
                (25, 35),
                0.9,
                2
            )

            draw_text(
                frame,
                f"Label: {current_label}",
                (25, 75),
                0.8,
                2
            )

            draw_text(
                frame,
                f"Samples: {counts[current_label]} / {SAMPLES_PER_LABEL}",
                (250, 75),
                0.7,
                2
            )

            # =============================================
            # TRẠNG THÁI TAY
            # =============================================

            if features is not None:

                draw_text(
                    frame,
                    "HAND DETECTED",
                    (25, h - 80),
                    0.7,
                    2
                )

                draw_text(
                    frame,
                    "Press SPACE to collect",
                    (25, h - 45),
                    0.6,
                    2
                )

            else:

                draw_text(
                    frame,
                    "NO HAND DETECTED",
                    (25, h - 80),
                    0.7,
                    2
                )

                draw_text(
                    frame,
                    "Place your hand in camera",
                    (25, h - 45),
                    0.6,
                    2
                )

            # =============================================
            # HƯỚNG DẪN
            # =============================================

            draw_text(
                frame,
                "SPACE: Save | N: Next | R: Reset | Q: Quit",
                (w - 500, h - 25),
                0.55,
                1
            )

            # =============================================
            # HIỂN THỊ
            # =============================================

            cv2.imshow(
                "SignBridge AI - Dataset Collector",
                frame
            )

            # =============================================
            # KEYBOARD
            # =============================================

            key = cv2.waitKey(1) & 0xFF

            # ---------------------------------------------
            # SPACE - LƯU SAMPLE
            # ---------------------------------------------

            if key == 32:

                if features is None:

                    print(
                        f"[{current_label}] "
                        "Không phát hiện bàn tay!"
                    )

                    continue

                # Tránh nhấn quá nhanh
                current_time = time.time()

                if current_time - last_save_time < 0.15:
                    continue

                last_save_time = current_time

                save_sample(
                    current_label,
                    features
                )

                counts[current_label] += 1

                print(
                    f"[{current_label}] "
                    f"Sample {counts[current_label]}/"
                    f"{SAMPLES_PER_LABEL}"
                )

                # Nếu đủ sample
                if counts[current_label] >= SAMPLES_PER_LABEL:

                    print()
                    print(
                        f"Đã hoàn thành label {current_label}"
                    )
                    print(
                        "Nhấn N để chuyển label tiếp theo."
                    )

            # ---------------------------------------------
            # N - LABEL TIẾP THEO
            # ---------------------------------------------

            elif key == ord("n"):

                if counts[current_label] < SAMPLES_PER_LABEL:

                    print()
                    print(
                        f"Label {current_label} "
                        f"chưa đủ {SAMPLES_PER_LABEL} mẫu."
                    )
                    print(
                        "Bạn vẫn có thể nhấn N để chuyển."
                    )

                current_index += 1

                if current_index >= len(LABELS):

                    print()
                    print("==============================")
                    print("ĐÃ ĐI QUA TOÀN BỘ A-Z")
                    print("==============================")

                    break

                current_label = LABELS[
                    current_index
                ]

                print()
                print(
                    "Chuyển sang label:",
                    current_label
                )
                print(
                    "Samples:",
                    counts[current_label]
                )

            # ---------------------------------------------
            # R - RESET
            # ---------------------------------------------

            elif key == ord("r"):

                print()
                print(
                    f"Reset label {current_label}..."
                )

                # Đọc toàn bộ dữ liệu
                rows = []

                with open(
                    CSV_FILE,
                    "r",
                    newline="",
                    encoding="utf-8"
                ) as file:

                    reader = csv.reader(file)

                    header = next(reader)

                    for row in reader:

                        if row[0] != current_label:
                            rows.append(row)

                # Ghi lại CSV
                with open(
                    CSV_FILE,
                    "w",
                    newline="",
                    encoding="utf-8"
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow(header)

                    writer.writerows(rows)

                counts[current_label] = 0

                print(
                    f"Label {current_label} "
                    "đã được reset."
                )

            # ---------------------------------------------
            # Q - THOÁT
            # ---------------------------------------------

            elif key == ord("q"):

                print()
                print("Đang thoát...")

                break

    finally:

        detector.close()

        cap.release()

        cv2.destroyAllWindows()

    print()
    print("====================================")
    print("DATASET COLLECTOR ĐÃ DỪNG")
    print("====================================")
    print(
        f"Dataset: {CSV_FILE}"
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()