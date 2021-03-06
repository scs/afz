#!/usr/bin/env python3
import traceback
import warnings

from algorithm.algorithm import Algorithm
from algorithm.counter.area_strategy import AreaCountingStrategy, AreaCountingStrategyWriter
from algorithm.counter.gate import Gate1, Gate2, Hallway, Stairs, GateWriterLayer, Gate1_960x540, Gate1_480x270, \
    Stairs_480x270, Stairs_960x540
from algorithm.object_counter import ObjectCounter, ObjectCounterWriterLayer, ObjectCounterCsvWriter
from algorithm.object_labeler import ObjectLabeler, ObjectLabelerWriterLayer
from algorithm.pickle import PickleLoader
from algorithm.readers import VideoReader, PickleReader
from algorithm.utils import PrintCounter
from algorithm.writers import ScreenDrawer, VideoWriter

from env import get_video_input_path, get_output_path, get_pickle_path, get_label_input_path

from logger import set_file_logger

# Remove tensorflow warnings
warnings.filterwarnings('ignore')


def run_counting_algo(filename, fps, strategy, offline):
    # tag = 'crowdhuman'
    # tag = 'score0_25'
    tag = 'tiny'
    output_path = get_output_path('algo-{}-{}fps-{}-{}'.format(filename, fps, tag, strategy.name))

    set_file_logger('{}.log'.format(output_path))

    video_reader = VideoReader(get_video_input_path(filename), fps)
    pickle_reader = PickleReader(pickle_path=get_pickle_path(filename, fps, tag=tag))

    algo = Algorithm(
        output_path=output_path,
        reader=pickle_reader if offline else video_reader,
        pipeline=[
            PickleLoader(pickle_path=get_pickle_path(filename, fps, tag=tag)),
            ObjectCounter(strategy=strategy),
            ObjectCounterCsvWriter(filename='{}-counting'.format(output_path)),
            AreaCountingStrategyWriter(filename='{}-strategy'.format(output_path), strategy=strategy),
            PrintCounter(),
            None if offline else VideoWriter(filename='{}+label+counting'.format(output_path),
                                             layers=[
                                                 GateWriterLayer(strategy.gate_region),
                                                 ObjectCounterWriterLayer(),
                                                 ObjectLabelerWriterLayer()
                                             ])
        ]
    )

    algo.run()


def run_all(offline=True):
    filenames_gate = [
        # ('041_gate1-1920x1080', Gate1),
        ('041_stairs-1920x1080', Stairs),
        # ('041_gate1-480x270', Gate1_480x270),
        # ('041_gate1-960x540', Gate1_960x540),
        # ('041_stairs-480x270', Stairs_480x270),
        # ('041_stairs-960x540', Stairs_960x540)
        # ('041_gate2-1920x1080', Gate2),
        # ('041_hallway-1920x1080', Hallway),
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
        # 10,
        # 15,
        30
    ]

    errors = []

    for filename, gate_region in filenames_gate:
        for f in fps:
            for strategy_type in strategies_type:
                try:
                    strategy = strategy_type(gate_region)
                    run_counting_algo(filename, f, strategy, offline=offline)
                except Exception as ex:
                    traceback.print_exc()
                    errors.append((filename, f, ex))

    print('Found {} errors:'.format(len(errors)))
    for filename, f, ex in errors:
        print('\t- {} at {}fps with {}'.format(filename, f, ex.__class__.__name__))


if __name__ == "__main__":
    # run_counting_algo('041_gate1-1920x1080', 15, AreaCountingStrategy(Gate1), offline=False)
    # run_counting_algo('041_gate2-1920x1080', 15, AreaCountingStrategy(Gate2), offline=False)
    # run_counting_algo('041_hallway-1920x1080', 15, AreaCountingStrategy(Hallway), offline=False)
    # run_counting_algo('041_stairs-1920x1080', 15, AreaCountingStrategy(Stairs), offline=False)

    # run_counting_algo('041_gate1-960x540', 10, AreaCountingStrategy(Gate1_960x540), offline=False)

    # run_counting_algo('041_stairs-960x540', 10, AreaCountingStrategy(Stairs_960x540), offline=False)

    run_all(offline=True)
