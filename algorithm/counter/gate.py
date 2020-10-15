import cv2
import numpy as np
from PIL import Image, ImageDraw

from algorithm.algorithm import AlgoContainer
from algorithm.writers import WriterLayer


def _cv3_find_contours(src):
    _, contours, _ = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _cv4_find_contours(src):
    contours, _ = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


class GateRegion(object):

    def __init__(self, inside, outside, factor=1.0, width=1920, height=1080):
        super().__init__()
        self.inside = self._scale(inside, factor)
        self.outside = self._scale(outside, factor)
        self.width = int(width * factor)
        self.height = int(height * factor)

        opencv_major = cv2.getVersionMajor()
        if opencv_major == 3:
            self._cv_find_contours = _cv3_find_contours
        elif opencv_major == 4:
            self._cv_find_contours = _cv4_find_contours
        else:
            raise ValueError('OpenCV {} is not supported'.format(cv2.getVersionMajor()))

        self.inside_contours = self._find_contours(self.inside, factor)
        self.outside_contours = self._find_contours(self.outside, factor)
        self.inside_mask = self._create_mask(self.inside, factor)
        self.outside_mask = self._create_mask(self.outside, factor)
        self.buffer_mask = np.logical_not(np.logical_or(self.inside_mask, self.outside_mask))

    def _scale(self, points, factor):
        def scale_point(p):
            x, y = p
            return int(factor * x), int(factor * y)

        return list(map(scale_point, points))

    def _find_contours(self, points, factor):
        src = np.zeros((self.height, self.width), dtype=np.uint8)

        n = len(points)
        for i in range(n):
            cv2.line(src, points[i], points[(i + 1) % n], (255), 10)

        contours = self._cv_find_contours(src)
        return contours[0]

    def _create_mask(self, points, factor):
        im = Image.fromarray(np.zeros((self.height, self.width), dtype=np.uint8))

        draw = ImageDraw.Draw(im)
        draw.polygon(points, 1)
        del draw

        return np.asarray(im)


class GateWriterLayer(WriterLayer):

    def __init__(self, gate_region: GateRegion):
        self.gate_region = gate_region

    def _draw_line(self, frame, points, color):
        n = len(points)
        for i in range(n):
            cv2.line(frame, points[i], points[(i + 1) % n], color, 10)

    def draw(self, container: AlgoContainer, frame):
        self._draw_line(frame, self.gate_region.inside, (0, 255, 0))
        self._draw_line(frame, self.gate_region.outside, (0, 0, 255))


Gate1 = GateRegion(
    inside=[
        (0, 0),
        (870, 0),
        (827, 886),
        (862, 1080),
        (0, 1080)
    ],
    outside=[
        (1920, 0),
        (1080, 0),
        (995, 677),
        (1100, 1080),
        (1920, 1080)
    ]
)

Gate1_960x540 = GateRegion(Gate1.inside, Gate1.outside, factor=0.5)
Gate1_480x270 = GateRegion(Gate1.inside, Gate1.outside, factor=0.25)

Stairs = GateRegion(
    inside=[
        (0, 430),
        (1300, 800),
        (1560, 1080),
        (1920, 1080),
        (1920, 0),
        (0, 0)
    ],
    outside=[
        (0, 560),
        (1300, 900),
        (1450, 1080),
        (0, 1080)
    ]
)

Stairs_960x540 = GateRegion(Stairs.inside, Stairs.outside, factor=0.5)
Stairs_480x270 = GateRegion(Stairs.inside, Stairs.outside, factor=0.25)

ScsCrop = GateRegion(
    inside=[
        (0, 0),
        (200, 0),
        (200, 480),
        (0, 480)
    ],
    outside=[
        (280, 0),
        (480, 0),
        (480, 480),
        (280, 480)
    ],
    width=480,
    height=480
)
