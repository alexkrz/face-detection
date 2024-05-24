from copy import deepcopy
from typing import Tuple

import cv2
import numpy as np
from jsonargparse import CLI


# Code adapted from https://github.com/sr6033/face-detection-with-OpenCV-and-DNN
def ssd_detect(
    img: np.ndarray,
    model_p: str = "checkpoints/ssd/ssd_facedetect.caffemodel",
    prototxt_p: str = "checkpoints/ssd/ssd_facedetect.prototxt.txt",
    conf_th: float = 0.5,
) -> Tuple[np.ndarray, int]:

    img = deepcopy(img)

    # Load serialized model from disk
    print("Loading model..")
    net = cv2.dnn.readNetFromCaffe(prototxt_p, model_p)

    # Convert image to input blob for neural net with input size 300 x 300
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image=cv2.resize(img, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
    )

    # pass the blob through the network and obtain the detections and
    # predictions
    print("Computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    n_detections = 0
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > conf_th:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated
            # probability
            text = f"{confidence * 100:.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            n_detections += 1

    return img, n_detections
