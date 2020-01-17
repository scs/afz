import typing
from abc import ABCMeta, abstractmethod

import cv2

from algorithm.algorithm import AlgoStep, AlgoContainer
from algorithm.counter.gate import GateRegion
from algorithm.writers import WriterLayer
from logger import get_logger


class CountingStrategy(metaclass=ABCMeta):

    def __init__(self, name: str, gate_region: GateRegion):
        self.logger = get_logger(self)
        self.name = name
        self.gate_region = gate_region

    @abstractmethod
    def is_inside(self, aabb, object_id):
        raise NotImplementedError()

    @abstractmethod
    def is_outside(self, aabb, object_id):
        raise NotImplementedError()

    @abstractmethod
    def is_buffer(self, aabb, object_id):
        raise NotImplementedError()


class ObjectCounterContainer(object):
    def __init__(self, object_id, aabb, strategy: CountingStrategy):
        self.logger = get_logger(self)
        self.object_id = object_id
        self.count_in = 0
        self.count_out = 0

        self.aabb = aabb

        self.is_tracked = True
        self.has_entered = False
        self.has_exited = False

        self.was_inside = strategy.is_inside(aabb, object_id)
        self.was_outside = strategy.is_outside(aabb, object_id)

    def pre_update(self):
        self.is_tracked = False
        self.has_entered = False
        self.has_exited = False

    def update(self, aabb, strategy: CountingStrategy):
        self.is_tracked = True

        self.aabb = aabb
        is_inside = strategy.is_inside(aabb, self.object_id)
        is_outside = strategy.is_outside(aabb, self.object_id)

        was_in_buffer = not self.was_outside and not self.was_inside
        if is_inside:
            self.was_inside = True
            if was_in_buffer or self.was_outside:
                self.logger.info('in for {}'.format(self.object_id))
                self.count_in += 1
                self.has_entered = True
                self.was_outside = False

        elif is_outside:
            self.was_outside = True
            if was_in_buffer or self.was_inside:
                self.logger.info('out for {}'.format(self.object_id))
                self.count_out += 1
                self.has_exited = True
                self.was_inside = False


class ObjectCounter(AlgoStep):
    def __init__(self, strategy):
        self.strategy = strategy
        self._counters: typing.Dict[int, ObjectCounterContainer] = {}

    def process(self, container: AlgoContainer):
        # FIXME Must be clone if we want to still use it later
        for object_id, counter in self._counters.items():
            counter.pre_update()

        for track in container.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            object_id = track.track_id
            aabb = track.to_tlbr()
            if object_id not in self._counters:
                self._counters[object_id] = ObjectCounterContainer(
                    object_id=object_id,
                    aabb=aabb,
                    strategy=self.strategy
                )
            else:
                self._counters[object_id].update(
                    aabb=aabb,
                    strategy=self.strategy
                )

        return container.extend_with(counts=self._counters)


class ObjectCounterWriterLayer(WriterLayer):

    def draw(self, container: AlgoContainer, frame):
        count_in = 0
        count_out = 0
        for object_id, counter in container.counts.items():
            count_in += counter.count_in
            count_out += counter.count_out

            if counter.is_tracked:
                color = (255, 0, 0)
                if counter.was_inside:
                    color = (0, 255, 0)

                elif counter.was_outside:
                    color = (0, 0, 255)

                x1, y1, x2, y2 = map(int, counter.aabb)
                cv2.putText(frame, str(object_id), (x1, y1 + 20), 0, 5e-3 * 200, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame, 'algo1', (10, 250), 0, 5e-3 * 200, (255, 128, 0), 2)
        cv2.putText(frame, str(count_in), (110, 250), 0, 5e-3 * 200, (128, 255, 128), 2)
        cv2.putText(frame, str(count_out), (170, 250), 0, 5e-3 * 200, (100, 100, 255), 2)


class ObjectCounterCsvWriter(AlgoStep):

    def __init__(self, filename):
        self.logger = get_logger(self)
        self.filename = filename

        self.writer = open('{}.csv'.format(self.filename), 'w')
        self.writer.write('frame,timestamp,entered,left\n')

    def process(self, container: AlgoContainer):
        count_in = 0
        count_out = 0
        for object_id, counter in container.counts.items():
            count_in += 1 if counter.has_entered else 0
            count_out += 1 if counter.has_exited else 0

        self.writer.write('{},{},{},{}\n'.format(container.position, container.time, count_in, count_out))

        return container
