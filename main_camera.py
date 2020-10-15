#!/usr/bin/env python3
import warnings

from algorithm.algorithm import Algorithm
from algorithm.counter.area_strategy import AreaCountingStrategy
from algorithm.counter.gate import GateWriterLayer, ScsCrop
from algorithm.object_counter import ObjectCounter, ObjectCounterWriterLayer
from algorithm.object_detector import ObjectDetector
from algorithm.object_labeler import ObjectLabelerWriterLayer, ObjectNoLabel
from algorithm.object_tracker import ObjectTracker
from algorithm.readers import CameraReader
from algorithm.utils import PrintCounter
from algorithm.writers import ScreenDrawer
from env import get_output_path
from logger import set_file_logger

# Remove tensorflow warnings
warnings.filterwarnings('ignore')


def run_camera(filename, strategy):
    tag = 'tiny'
    output_path = get_output_path('camera-{}-{}-{}'.format(filename, tag, strategy.name))

    set_file_logger('{}.log'.format(output_path))

    algo = Algorithm(
        output_path=output_path,
        reader=CameraReader(),
        pipeline=[
            ObjectNoLabel(),
            ObjectDetector(score=0.25),
            ObjectTracker(max_age=10),
            ObjectCounter(strategy=strategy),
            PrintCounter(),
            ScreenDrawer(filename='{}+counting'.format(output_path),
                         layers=[
                             GateWriterLayer(strategy.gate_region),
                             ObjectCounterWriterLayer(),
                             ObjectLabelerWriterLayer()
                         ])
        ]
    )
    algo.run()


if __name__ == "__main__":
    strategy = AreaCountingStrategy(ScsCrop)
    run_camera('scs-gang', strategy)
