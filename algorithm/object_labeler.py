from functools import reduce

import cv2

from algorithm.algorithm import AlgoStep, AlgoContainer
from algorithm.writers import WriterLayer
from logger import get_logger


class ObjectLabel(object):
    def __init__(self, position, timestamp, count_in, count_out):
        self.position = position
        self.timestamp = timestamp
        self.count_in = count_in
        self.count_out = count_out

    def merge(self, other):
        return ObjectLabel(
            position=other.position,
            timestamp=other.timestamp,
            count_in=self.count_in + other.count_in,
            count_out=self.count_out + other.count_out
        )


class ObjectNoLabel(AlgoStep):
    def process(self, container: AlgoContainer):
        return container.extend_with(label=ObjectLabel(
            position=container.position,
            timestamp=container.time,
            count_in=0,
            count_out=0
        ))


class ObjectLabeler(AlgoStep):
    def __init__(self, label_path):
        self.logger = get_logger(self)
        self.labels = self._retrieve_labels(label_path)
        self.logger.info('loaded {} labels'.format(len(self.labels)))

    def _retrieve_labels(self, label_path):
        with open(label_path, 'r') as f:
            # Skip first line
            f.readline()

            def to_label(line):
                position, timestamp, count_in, count_out = line.strip().split(',')
                return ObjectLabel(
                    position=int(position),
                    timestamp=float(timestamp),
                    count_in=int(count_in),
                    count_out=int(count_out)
                )

            return list(map(to_label, f))

    def process(self, container: AlgoContainer):
        labels_till_now = list(filter(lambda label: label.position < container.position, self.labels))
        self.logger.info('found {} labels till position {}'.format(len(labels_till_now), container.position))
        label = reduce(lambda x, y: x.merge(y), labels_till_now)
        return container.extend_with(label=label)


class ObjectLabelerWriterLayer(WriterLayer):

    def draw(self, container: AlgoContainer, frame):
        cv2.putText(frame, 'in', (110, 150), 0, 5e-3 * 200, (128, 255, 128), 2)
        cv2.putText(frame, 'out', (170, 150), 0, 5e-3 * 200, (100, 100, 255), 2)

        cv2.putText(frame, 'ref', (10, 200), 0, 5e-3 * 200, (255, 128, 0), 2)
        cv2.putText(frame, str(container.label.count_in), (110, 200), 0, 5e-3 * 200, (128, 255, 128), 2)
        cv2.putText(frame, str(container.label.count_out), (170, 200), 0, 5e-3 * 200, (100, 100, 255), 2)
