from copy import deepcopy

import cv2
import numpy as np


def draw_bboxes_and_keypoints(img: np.ndarray, bboxes: list, keypoints_all: list) -> np.ndarray:
    img = deepcopy(img)
    for i in range(len(bboxes)):
        # bbox is a list of four integers (x, y, w, h)
        bbox = bboxes[i]
        x, y, w, h = bbox
        # keypoints is a dictionary of the five facial landmarks with their corresponding (x, y) coordinates
        keypoints = keypoints_all[i]

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 155, 255),
            2,
        )

        cv2.circle(img, (keypoints["left_eye"]), 2, (0, 155, 255), 2)
        cv2.circle(img, (keypoints["right_eye"]), 2, (0, 155, 255), 2)
        cv2.circle(img, (keypoints["nose"]), 2, (0, 155, 255), 2)
        cv2.circle(img, (keypoints["mouth_left"]), 2, (0, 155, 255), 2)
        cv2.circle(img, (keypoints["mouth_right"]), 2, (0, 155, 255), 2)
    return img
