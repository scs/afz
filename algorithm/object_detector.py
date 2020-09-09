import cv2
from PIL import Image

from algorithm.writers import WriterLayer
from yolo3.yolo import YOLO

from algorithm.algorithm import AlgoStep, AlgoContainer
from logger import get_logger


class ObjectDetection(object):
    # TODO
    pass


class DataSet(object):
    normal = 'yolov3'
    tiny = 'tiny'
    crowdhuman = 'crowdhuman'


class ObjectDetector(AlgoStep):
    def __init__(self, score=0.25, dataset=DataSet.normal):
        self.logger = get_logger(self)
        if dataset == DataSet.normal:
            self.yolo = YOLO(
                score=score
            )
        elif dataset == DataSet.tiny:
            self.yolo = YOLO(
                score=score,
                model_path='model_data/yolov3-tiny.h5',
                anchors_path='model_data/yolov3-tiny_anchors.txt'
            )
        elif dataset == DataSet.crowdhuman:
            self.yolo = YOLO(
                score=score,
                model_path='model_data/yolov3-tiny-crowdhuman-80000.h5',
                anchors_path='model_data/yolov3-tiny-crowdhuman_anchors.txt',
                classes_path='model_data/crowdhuman.names'
            )
        else:
            raise ValueError('Dataset {} unknown'.format(dataset))

    def process(self, container: AlgoContainer):
        image = Image.fromarray(container.frame.image[..., ::-1])  # bgr to rgb
        detections = self.yolo.detect_image(image)
        if not detections:
            detections = []
        # TODO Transform detections into ValueObject
        return container.extend_with(detections=detections)


class ObjectDetectorWriterLayer(WriterLayer):

    def draw(self, container: AlgoContainer, frame):
        for bbox in container.detections:
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = x1 + int(bbox[2])
            y2 = y1 + int(bbox[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
