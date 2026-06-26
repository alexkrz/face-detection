import cv2
import numpy as np


def non_max_suppression_onnx(
    preds: np.ndarray,
    conf_thres: float = 0.25,
    iou_thresh: float = 0.45,
) -> np.ndarray:
    """Non-Maximum Suppression for ONNX predictions.

    Args:
        preds: Predictions [16, n_predictions] containing:
               [0:4]   - bbox (x_center, y_center, width, height)
               [4]     - objectness confidence
               [5:15]  - landmarks (5 keypoints: left_eye, right_eye, nose, left_mouth, right_mouth)
               [15]    - class confidence
        conf_thres: Confidence threshold
        iou_thresh: IoU threshold for NMS

    Returns:
        Detections [n_det, 16] containing:
        [0:4]   - bbox (x1, y1, x2, y2)
        [4]     - confidence (objectness * class)
        [5:15]  - landmarks (5 keypoints: left_eye, right_eye, nose, left_mouth, right_mouth)
        [15]    - class index
    """
    # Filter by objectness confidence (preds[4])
    obj_conf_mask = preds[4] > conf_thres
    detections = preds[:, obj_conf_mask]

    # Multiply objectness confidence by class confidence (detections[15:])
    detections[15:] *= detections[4:5]

    # Get best class confidence and index
    class_conf = detections[15:].max(axis=0, keepdims=True)
    class_idx = detections[15:].argmax(axis=0, keepdims=True).astype(float)

    # Convert bbox from [x_center, y_center, width, height] to [x1, y1, x2, y2]
    bbox_xywh = detections[:4].T
    center_xy, size_wh = bbox_xywh[:, :2], bbox_xywh[:, 2:4]
    bbox_xyxy = np.concatenate([center_xy - size_wh / 2, center_xy + size_wh / 2], axis=1)

    # Concatenate: [bbox_xyxy, confidence, landmarks, class_idx]
    landmarks = detections[5:15].T
    detections = np.concatenate([bbox_xyxy, class_conf.T, landmarks, class_idx.T], axis=1)

    # Filter by class confidence
    detections = detections[class_conf.flatten() > conf_thres]

    if len(detections) == 0:
        return np.zeros((0, 16))

    # NMS - keep boxes with IoU below threshold
    boxes = detections[:, :4]
    scores = detections[:, 4]
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep_indices = []
    while order.size > 0:
        idx = order[0]
        keep_indices.append(idx)

        # Compute IoU between current box and remaining boxes
        inter_w = np.maximum(0, np.minimum(x2[idx], x2[order[1:]]) - np.maximum(x1[idx], x1[order[1:]]))
        inter_h = np.maximum(0, np.minimum(y2[idx], y2[order[1:]]) - np.maximum(y1[idx], y1[order[1:]]))
        inter_area = inter_w * inter_h
        iou = inter_area / (areas[idx] + areas[order[1:]] - inter_area)

        # Keep boxes with IoU <= threshold
        order = order[np.concatenate([[0], np.where(iou <= iou_thresh)[0] + 1])][1:]

    return detections[np.array(keep_indices, dtype=int)]


def rescale_coordinates(
    preds: np.ndarray,
    img: np.ndarray,
) -> np.ndarray:
    # Rescale coordinates to original image
    orig_h, orig_w = img.shape[:2]
    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0

    # Scale bounding boxes (x1, y1, x2, y2)
    preds[:, 0] *= scale_x  # x1
    preds[:, 1] *= scale_y  # y1
    preds[:, 2] *= scale_x  # x2
    preds[:, 3] *= scale_y  # y2

    # Scale landmarks (5 landmarks with x, y coordinates starting at index 5)
    for i in range(5):
        preds[:, 5 + 2 * i] *= scale_x  # landmark x
        preds[:, 5 + 2 * i + 1] *= scale_y  # landmark y

    return preds


def adjust_boxes_and_kpts(boxes: np.ndarray, kpts: np.ndarray) -> tuple[list, list]:
    bboxes = []
    keypoints_all = []
    order = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
    for i in range(len(boxes)):
        box = boxes[i]
        kp = kpts[i]
        x1, y1, x2, y2 = box.astype(int)
        w = x2 - x1
        h = y2 - y1
        bbox = [x1, y1, w, h]
        bboxes.append(bbox)

        keypoints = {}
        for j in range(len(order)):
            # Question: Why does every keypoint have a third value?
            # -> Third value could be keypoint confidence
            keypoints[order[j]] = (int(kp[j * 2]), int(kp[j * 2 + 1]))
        keypoints_all.append(keypoints)
    return bboxes, keypoints_all


def yolov5_detect(
    img: np.ndarray,
    checkpoint_p: str = "checkpoints/yolo/yolov5n-face.onnx",
) -> tuple[list, list]:
    """Detect faces with YOLO network

    Args:
        img (np.ndarray): Input image in OpenCV BGR format.
        ckpt_root_dir (str, optional): The checkpoint directory.

    Returns:
        Tuple[List, List]: Bounding boxes and keypoints
    """

    model = cv2.dnn.readNetFromONNX(checkpoint_p)

    # Preprocess image for ONNX model
    blob = cv2.dnn.blobFromImage(
        img,
        1 / 255.0,
        (640, 640),
        swapRB=True,
        crop=False,
    )

    # Set input and run inference
    model.setInput(blob)
    outputs = model.forward()
    preds_raw = outputs[0]

    # Apply NMS
    preds = non_max_suppression_onnx(preds_raw, conf_thres=0.25, iou_thresh=0.45)

    # Rescale coordinates
    preds = rescale_coordinates(preds, img)
    boxes = np.array(preds[:, :4], dtype=int)
    keypoints = np.array(preds[:, 5:15], dtype=int)
    bboxes, keypoints_all = adjust_boxes_and_kpts(boxes, keypoints)

    return bboxes, keypoints_all
