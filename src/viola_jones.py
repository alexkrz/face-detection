from copy import deepcopy
from typing import Tuple

import cv2
import numpy as np


def viola_jones_detect(
    img: np.ndarray,
    checkpoint_p: str = "checkpoints/viola_jones/haarcascade_frontalface_alt.xml",
) -> Tuple[np.ndarray, int]:
    """Applies Viola-Jones detection to input image
    Paper: https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf

    Args:
        img (np.ndarray): Numpy array with shape (H, W, C) in RGB color ordering

    Returns:
        img (np.ndarray): The modified input image with detections drawn as rectangles
        n_detections (int): Number of detections
    """
    img = deepcopy(img)
    frame_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    face_cascade = cv2.CascadeClassifier()
    face_cascade.load(checkpoint_p)

    faces = face_cascade.detectMultiScale(frame_gray)
    n_detections = len(faces)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return img, n_detections
