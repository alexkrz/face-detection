# Face Detection Knowledge

List of important approaches:

1. [Viola-Jones Detector](https://ieeexplore.ieee.org/document/990517) (CVPR, 2001)
    - Available in OpenCV

2. [MTCNN](https://ieeexplore.ieee.org/document/7553523) (2016)
    - Use OpenCV DNN implementation from <https://github.com/ksachdeva/opencv-mtcnn>
    - The Model was originally trained in Caffe

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

### Todos

[ ] How do I load a Caffe model with the OpenCV DNN library in Python?
