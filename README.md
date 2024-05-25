# Face Detection Knowledge

List of important approaches:

1. [Viola-Jones Detector](https://ieeexplore.ieee.org/document/990517) (CVPR, 2001)
    - Available in OpenCV
    - A tutorial can be found here: <https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html>

2. [MTCNN](https://ieeexplore.ieee.org/document/7553523) (2016)
    - Original code: <https://github.com/kpzhang93/MTCNN_face_detection_alignment>
    - The model was originally developed in Caffe
    - Repository with OpenCV and C++: <https://github.com/ksachdeva/opencv-mtcnn>
    - Repository with OpenCV and Python: <https://github.com/linxiaohui/mtcnn-opencv>

3. [SSD](http://arxiv.org/abs/1512.02325) (ECCV, 2016)
    - The official code repository is here: <https://github.com/weiliu89/caffe/tree/ssd>
    - OpenCV integrates a pretrained model that can be found here: <https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector>
    - Instructions how the SSD model was trained are also in the OpenCV repository
    - The weights probably stem from this repository: <https://github.com/sr6033/face-detection-with-OpenCV-and-DNN>

4. [RetinaFace](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html) (CVPR, 2020)
    - RetinaFace is originally trained in MXNet and the code can be found here: <https://github.com/deepinsight/insightface/tree/master/detection/retinaface>
    - For usage in inference, I would use this repository: <https://github.com/Charrin/RetinaFace-Cpp>
    - The Caffe model can probably be loaded with OpenCV

5. [YOLO5-Face](http://arxiv.org/abs/2105.12931) (ArXiV, 2022)
    - The official code repository is here: <https://github.com/deepcam-cn/yolov5-face>
    - The model is trained on WiderFace dataset with Pytorch
    - The YoloV8-Face repository is here: <https://github.com/derronqi/yolov8-face/tree/main>
    - Converted ONNX models for yolov8 can be found here: <https://github.com/hpc203/yolov8-face-landmarks-opencv-dnn>

## Set up repository

1. Install miniconda
2. Create conda environment with

    ```bash
    conda env create -n fdetect -f environment.yml
    ```

3. Install pip requirements

    ```bash
    conda activate fdetect
    pip install -r requirements.txt
    ```

4. Install pre-commit

    ```bash
    pre-commit install
    ```

5. Perform face detection

    ```bash
    python detect_faces.py (-h)
    ```

    Use `-h` flag to show options

## Todos

- [ ] Fix performance discrepancy between `src/mtcnn_onnx.py` and `src/mtcnn_caffe.py`
