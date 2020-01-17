import cv2
import numpy as np

from algorithm.writers import WriterLayer
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

from algorithm.algorithm import AlgoStep, AlgoContainer
from logger import get_logger


class ObjectTracker(AlgoStep):
    def __init__(self, max_age=30):
        self.logger = get_logger(self)

        max_cosine_distance = 0.3
        nn_budget = None
        self.nms_max_overlap = 1.0

        model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_age=max_age)

    def process(self, container: AlgoContainer):
        features = self.encoder(container.frame.image, container.detections)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(container.detections, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        return container.extend_with(tracks=self.tracker.tracks)


class ObjectTrackerWriterLayer(WriterLayer):

    def draw(self, container: AlgoContainer, frame):
        for track in container.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(frame, str(track.track_id), (x1, y1 + 20), 0, 5e-3 * 200, (0, 255, 0), 2)
