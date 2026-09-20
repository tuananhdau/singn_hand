import numpy as np


def extract_landmarks(results):
    """
    Lấy 21 landmark của bàn tay đầu tiên.
    Trả về vector gồm 63 giá trị.
    """

    if not results.hand_landmarks:
        return None

    hand = results.hand_landmarks[0]

    # Landmark cổ tay
    wrist = hand[0]

    features = []

    for landmark in hand:

        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        features.extend([x, y, z])

    return np.array(features, dtype=np.float32)