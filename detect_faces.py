import cv2
import numpy as np
from jsonargparse import CLI

from src.ssd import ssd_detect
from src.viola_jones import viola_jones_detect

func_dict = {
    "ssd": ssd_detect,
    "viola_jones": viola_jones_detect,
}


def main(
    img_p: str = "data/img1.jpg",
    method_name: str = "ssd",
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
