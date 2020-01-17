#!/usr/bin/env python3
import traceback
import warnings

from algorithm.algorithm import Algorithm
from algorithm.counter.area_strategy import AreaCountingStrategy, AreaCountingStrategyWriter
from algorithm.counter.gate import Gate1, Stairs, Gate1_480x270, Gate1_960x540, Stairs_480x270, Stairs_960x540, Gate2, \
    Hallway, GateWriterLayer
from algorithm.object_counter import ObjectCounter, ObjectCounterCsvWriter, ObjectCounterWriterLayer
from algorithm.object_detector import ObjectDetector, DataSet
from algorithm.object_labeler import ObjectLabeler, ObjectLabelerWriterLayer, ObjectNoLabel
from algorithm.object_tracker import ObjectTracker
from algorithm.pickle import PickleWriter
from algorithm.readers import VideoReader
from algorithm.utils import PrintCounter, ScaleFrame
from algorithm.writers import VideoWriter
from env import get_output_path, get_video_input_path, get_label_input_path, get_pickle_path

from logger import set_file_logger

# Remove tensorflow warnings
warnings.filterwarnings('ignore')


def run_algo(filename, fps, strategy):
    dataset = DataSet.normal
    tag = '{}'.format(dataset)
    output_path = get_output_path('algo-{}-{}fps-{}-{}'.format(filename, fps, tag, strategy.name))

    set_file_logger('{}.log'.format(output_path))

    algo = Algorithm(
        output_path=output_path,
        reader=VideoReader(get_video_input_path(filename), fps),
        pipeline=[
            ScaleFrame(scale=0.25),
            ObjectLabeler(get_label_input_path(filename)),
            ObjectDetector(score=0.25, dataset=dataset),
            ObjectTracker(max_age=10),
            PickleWriter(pickle_path=get_pickle_path(filename, fps, tag=tag)),
            ObjectCounter(strategy=strategy),
            ObjectCounterCsvWriter(filename='{}-counting'.format(output_path)),
            AreaCountingStrategyWriter(filename='{}-strategy'.format(output_path), strategy=strategy),
            PrintCounter(),
            VideoWriter(filename='{}+label+counting'.format(output_path),
                        layers=[
                            GateWriterLayer(strategy.gate_region),
                            ObjectCounterWriterLayer(),
                            ObjectLabelerWriterLayer()
                        ])
        ]
    )
    algo.run()


def run_all():
    filenames_gate = [
        # ('041_gate1-1920x1080', Gate1),
        # ('041_gate1-480x270', Gate1_480x270),
        # ('041_gate1-960x540', Gate1_960x540),
        # ('041_stairs-1920x1080', Stairs),
        # ('041_stairs-480x270', Stairs_480x270),
        # ('041_stairs-960x540', Stairs_960x540),
        # ('041_gate2-1920x1080', Gate2),
        # ('041_hallway-1920x1080', Hallway),
        ('042_Aufnahmen Treppe', Stairs_480x270),
        ('042_Aufnahmen Tür 1', Gate1_480x270),
    ]

    strategies_type = [
        # TopPointCountingStrategy,
        # CenterPointCountingStrategy,
        # BottomPointCountingStrategy,
        AreaCountingStrategy
    ]

    fps = [
        # 1,
        # 3,
        # 5,
        10,
        # 15,
        # 30
    ]

    errors = []

    for filename, gate_region in filenames_gate:
        for f in fps:
            for strategy_type in strategies_type:
                try:
                    strategy = strategy_type(gate_region)
                    run_algo(filename, f, strategy)
                except Exception as ex:
                    traceback.print_exc()
                    errors.append((filename, f, ex))

    print('Found {} errors:'.format(len(errors)))
    for filename, f, ex in errors:
        print('\t- {} at {}fps with {}'.format(filename, f, ex.__class__.__name__))


if __name__ == "__main__":
    # run_algo('041_stairs-1920x1080', 15, AreaCountingStrategy(Stairs))
    run_all()
