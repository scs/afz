#!/usr/bin/env python3
import traceback
import warnings

from algorithm.algorithm import Algorithm
from algorithm.object_detector import ObjectDetector, ObjectDetectorWriterLayer, DataSet
from algorithm.object_labeler import ObjectLabeler, ObjectLabelerWriterLayer
from algorithm.object_tracker import ObjectTracker, ObjectTrackerWriterLayer
from algorithm.pickle import PickleWriter
from algorithm.readers import VideoReader
from algorithm.writers import VideoWriter
from env import get_output_path, get_video_input_path, get_label_input_path, get_pickle_path
from logger import set_file_logger

# Remove tensorflow warnings
warnings.filterwarnings('ignore')


def run_tracking_algo(filename, fps):
    tag = 'tiny'
    output_path = get_output_path('algo-{}-{}fps-{}'.format(filename, fps, tag))

    set_file_logger('{}.log'.format(output_path))

    algo = Algorithm(
        output_path=output_path,
        reader=VideoReader(get_video_input_path(filename), fps),
        pipeline=[
            ObjectLabeler(get_label_input_path(filename)),
            ObjectDetector(score=0.25, dataset=DataSet.tiny),
            ObjectTracker(max_age=10),
            PickleWriter(pickle_path=get_pickle_path(filename, fps, tag=tag)),
            VideoWriter(filename='{}+detection+tracking+label'.format(output_path),
                        layers=[
                            ObjectDetectorWriterLayer(),
                            ObjectTrackerWriterLayer(),
                            ObjectLabelerWriterLayer()
                        ])
        ]
    )
    algo.run()


def run_all():
    filenames = [
        '041_stairs-1920x1080',
    ]

    fps = [
        15,
        30
    ]

    errors = []

    for filename in filenames:
        for f in fps:
            try:
                run_tracking_algo(filename, f)
            except Exception as ex:
                traceback.print_exc()
                errors.append((filename, f, ex))

    print('Found {} errors:'.format(len(errors)))
    for filename, f, ex in errors:
        print('\t- {} at {}fps with {}'.format(filename, f, ex.__class__.__name__))


if __name__ == "__main__":
    run_all()
