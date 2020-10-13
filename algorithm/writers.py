from abc import ABCMeta, abstractmethod
from typing import List

from timeit import time

import cv2

from algorithm.algorithm import AlgoStep, AlgoContainer
from logger import get_logger


class WriterLayer(metaclass=ABCMeta):

    @abstractmethod
    def draw(self, container: AlgoContainer, frame):
        raise NotImplementedError()


class VideoWriter(AlgoStep):

    def __init__(self, filename, layers: List[WriterLayer]):
        self.filename = filename
        self.writer = None
        self.layers = layers

    def _create_or_get_writer(self, container: AlgoContainer):
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            fps = container.frame.fps
            wh = (container.frame.width, container.frame.height)
            self.writer = cv2.VideoWriter('{}.avi'.format(self.filename), fourcc, fps, wh)

        return self.writer

    def process(self, container: AlgoContainer):
        writer = self._create_or_get_writer(container)

        cv_frame = container.frame.image.copy()
        for layer in self.layers:
            layer.draw(container, cv_frame)

        writer.write(cv_frame)
        return container


class ScreenDrawer(AlgoStep):

    def __init__(self, filename, layers: List[WriterLayer]):
        self.logger = get_logger(self)
        self.filename = filename
        self.layers = layers

        self.last_video_time = 0
        self.last_local_time = time.time()

    def process(self, container: AlgoContainer):
        current_time = time.time()
        delta_local = 1000 * (current_time - self.last_local_time)
        delta_video = container.time - self.last_video_time

        self.logger.info('{} <? {}'.format(delta_local, delta_video))

        if delta_local < delta_video:
            frame = container.frame.image.copy()
            for layer in self.layers:
                layer.draw(container, frame)

            frame2 = cv2.flip(frame, 1)
            height = frame.shape[0]
            width = frame.shape[1]
            dim = (width * 2, height * 2)
            frame3 = cv2.resize(frame2, dim, cv2.INTER_AREA)
            cv2.imshow(self.filename, frame3)
            cv2.waitKey(int(delta_video - delta_local + 1))
            # TODO Read key and save image when p

        self.last_video_time = container.time
        self.last_local_time = current_time
        return container


class CsvWriter(AlgoStep):

    # def __init__(self, filename):
    #     self.writer = open('{}.csv'.format(filename), 'w')
    #     self.writer.write('{}\n'.format(','.join([
    #         'timestamp',
    #         'id',
    #         'state',
    #         'time_since_update',
    #         'hits',
    #         'age',
    #         'x1',
    #         'y1',
    #         'x2',
    #         'y2'
    #     ])))

    def process(self, container: AlgoContainer):

        if container.tracks:
            for track in container.tracks:
                bbox = track.to_tlbr()
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])

                # self.writer.write('{}\n'.format(','.join(map(str, [
                #     container.time,
                #     track.track_id,
                #     track.state,
                #     track.time_since_update,
                #     track.hits,
                #     track.age,
                #     x1,
                #     y1,
                #     x2,
                #     y2
                # ]))))

        return container
