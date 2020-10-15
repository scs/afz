import os
import pickle
from timeit import time

import cv2

from algorithm.algorithm import AlgoReader, AlgoContainer, AlgoFrame
from logger import get_logger


class VideoReader(AlgoReader):

    def __init__(self, filename, fps):
        self.logger = get_logger(self)
        self.filename = filename
        self.video_capture = cv2.VideoCapture(self.filename)
        self.fps = fps
        video_fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))

        if video_fps % self.fps > 0:
            raise ValueError('target_fps {} should be a divider of video_fps {}'.format(self.fps, video_fps))

        self.skip = int(video_fps / self.fps) - 1

    def has_more(self):
        for _ in range(self.skip):
            self.video_capture.grab()

        return self.video_capture.grab()

    def read_next(self):
        ret, image = self.video_capture.retrieve()

        position = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        time = float(self.video_capture.get(cv2.CAP_PROP_POS_MSEC))
        width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return AlgoContainer(
            position=position,
            time=time,
            frame=AlgoFrame(
                width=width,
                height=height,
                image=image,
                fps=self.fps
            )
        )


class CameraReader(AlgoReader):

    def __init__(self):
        self.logger = get_logger(self)
        self.video_capture = cv2.VideoCapture(0)
        self._t0 = time.time()

    def _get_fps(self):
        t1 = time.time()
        fps = 1.0 / (1000 * (t1 - self._t0))
        self._t0 = t1
        return fps

    def has_more(self):
        return True

    def read_next(self):
        ret, image = self.video_capture.read()

        position = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        time = float(self.video_capture.get(cv2.CAP_PROP_POS_MSEC))
        width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        image = image[0:480, 0:480]
        image = cv2.flip(image, 1)

        return AlgoContainer(
            position=position,
            time=time,
            frame=AlgoFrame(
                width=width - 160,
                height=height,
                image=image,
                fps=8
            )
        )


class PickleReader(AlgoReader):
    def __init__(self, pickle_path):
        self.logger = get_logger(self)
        self.pickle_path = pickle_path

        self.timestamps = self._retrieve_timestamps(self.pickle_path)
        self.logger.info('found {} timestamps'.format(len(self.timestamps)))

    def _retrieve_timestamps(self, pickle_path):
        def is_pickle(filename: str):
            return filename.endswith('.pickle')

        def to_timestamp(filename: str):
            return int(filename.replace('.pickle', ''))

        files = os.listdir(pickle_path)
        return list(sorted(map(to_timestamp, filter(is_pickle, files))))

    def has_more(self):
        return len(self.timestamps) > 0

    def read_next(self):
        timestamp = self.timestamps.pop(0)
        with open(os.path.join(self.pickle_path, '{}.pickle'.format(timestamp)), 'rb') as f:
            return pickle.load(f)
