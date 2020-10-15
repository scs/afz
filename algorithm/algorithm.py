import sys
import traceback
import typing
from abc import ABCMeta, abstractmethod
from timeit import time

from logger import get_logger


class AlgoFrame(object):
    def __init__(self, width, height, image, fps):
        self.width = width
        self.height = height
        self.image = image
        self.fps = fps


class AlgoContainer(object):
    def __init__(self,
                 position,
                 time,
                 frame: AlgoFrame = None,
                 label=None,
                 detections=None,
                 tracks=None,
                 counts=None):
        self.position = position
        self.time = time
        self.frame = frame
        self.label = label
        self.detections = detections
        self.tracks = tracks
        self.counts = counts

    def extend_with(self, frame=None, label=None, detections=None, tracks=None, counts=None):
        return AlgoContainer(
            position=self.position,
            time=self.time,
            frame=frame if frame is not None else self.frame,
            label=label if label is not None else self.label,
            detections=detections if detections is not None else self.detections,
            tracks=tracks if tracks is not None else self.tracks,
            counts=counts if counts is not None else self.counts
        )

    def without_frame(self):
        return AlgoContainer(
            position=self.position,
            time=self.time,
            label=self.label,
            detections=self.detections,
            tracks=self.tracks,
            counts=self.counts
        )


class AlgoReader(metaclass=ABCMeta):

    @abstractmethod
    def has_more(self):
        raise NotImplementedError()

    @abstractmethod
    def read_next(self):
        raise NotImplementedError()


class AlgoStep(metaclass=ABCMeta):
    @abstractmethod
    def process(self, container: AlgoContainer):
        raise NotImplementedError()


class Algorithm(object):
    def __init__(self, output_path, reader: AlgoReader, pipeline: typing.List[AlgoStep]):
        self.logger = get_logger(self)
        self.reader = reader
        self.pipeline = [p for p in pipeline if p is not None]
        self.fps = 0
        self.writer = open('{}-stats.csv'.format(output_path), 'w')

    def run_frame(self):
        container = self.reader.read_next()
        self.logger.info('Starting processing frame at time {}ms...'.format(container.time))

        self.writer.write('{},'.format(container.position))
        self.writer.write('{},'.format(container.time))

        t1 = time.time()
        for step in self.pipeline:
            try:
                st1 = time.time()
                container = step.process(container)
                st2 = time.time()
                delta_st = 1000 * (st2 - st1)
                self.logger.info('Timed step {} at {}ms'.format(step.__class__.__name__, delta_st))
                self.writer.write('{},'.format(delta_st))
            except Exception:
                self.logger.error('Error while processing {} at time {}ms'.format(step, container.time))
                traceback.print_exc(file=sys.stdout)
                return None

        t2 = time.time()
        delta_t = 1000 * (t2 - t1)
        self.logger.info('processed frame {} in {}ms'.format(container.time, delta_t))
        self.writer.write('{}\n'.format(delta_t))

        self.fps = (self.fps + (1. / (time.time() - t1))) / 2
        self.logger.info('running at {} fps'.format(self.fps))

        return container

    def run(self):
        names = list(map(lambda step: step.__class__.__name__, self.pipeline))
        self.writer.write('position,timestamp,{},total\n'.format(','.join(names)))
        container = None
        while self.reader.has_more():
            container = self.run_frame()

        return container
