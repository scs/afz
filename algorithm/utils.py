import cv2

from algorithm.algorithm import AlgoStep, AlgoContainer, AlgoFrame
from logger import get_logger


class ScaleFrame(AlgoStep):

    def __init__(self, scale):
        self.logger = get_logger(self)
        self.scale = scale

    def _scale(self, frame: AlgoFrame):
        width = int(frame.image.shape[1] * self.scale)
        height = int(frame.image.shape[0] * self.scale)
        dim = (width, height)
        return AlgoFrame(
            width=width,
            height=height,
            image=cv2.resize(frame.image, dim, interpolation=cv2.INTER_AREA),
            fps=frame.fps
        )

    def process(self, container: AlgoContainer):
        return container.extend_with(
            frame=self._scale(container.frame)
        )


class PrintCounter(AlgoStep):

    def __init__(self):
        self.logger = get_logger(self)

    def process(self, container: AlgoContainer):
        count_in = 0
        count_out = 0
        for object_id, counter in container.counts.items():
            count_in += counter.count_in
            count_out += counter.count_out

        self.logger.info('count_in = {} vs ref = {}'.format(count_in, container.label.count_in))
        self.logger.info('count_out = {} vs ref = {}'.format(count_out, container.label.count_out))

        return container
