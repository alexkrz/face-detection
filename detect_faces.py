import cv2
import numpy as np
from jsonargparse import CLI

from src.mtcnn_caffe import mtcnn_caffe_detect
from src.mtcnn_onnx import mtcnn_onnx_detect
from src.ssd import ssd_detect
from src.viola_jones import viola_jones_detect
from src.yolo import yolo_detect

func_dict = {
    "ssd": ssd_detect,
    "viola_jones": viola_jones_detect,
    "yolo": yolo_detect,
    "mtcnn_caffe": mtcnn_caffe_detect,
    "mtcnn_onnx": mtcnn_onnx_detect,
}


def main(
    img_p: str = "data/img2.jpg",
    method_name: str = "mtcnn_onnx",
):
    assert method_name in func_dict.keys()
    img_in = cv2.imread(img_p)
    cv2.imshow("Input Image", img_in)

    img_out, n_detections = func_dict[method_name](img_in)
    print("Number of detections:", n_detections)

    cv2.imshow("Output Image", img_out)
    cv2.waitKey(0)


if __name__ == "__main__":
    CLI(main, as_positional=False)
