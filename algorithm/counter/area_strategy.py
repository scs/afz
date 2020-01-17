import numpy as np

from algorithm.algorithm import AlgoStep, AlgoContainer
from algorithm.counter.gate import GateRegion
from algorithm.object_counter import CountingStrategy
from logger import get_logger


class AreaCountingStrategy(CountingStrategy):

    def __init__(self, gate_region: GateRegion):
        super().__init__(
            name='area',
            gate_region=gate_region
        )

    def compute_intersection(self, aabb, mask_region):
        h, w = mask_region.shape

        x1, y1, x2, y2 = aabb
        x1 = int(max(x1, 0))
        x2 = int(min(x2, w))
        y1 = int(max(y1, 0))
        y2 = int(min(y2, h))

        aabb_region = mask_region[y1:y2, x1:x2]
        area_inter = np.sum(aabb_region)

        area_aabb = aabb_region.size

        return area_aabb, area_inter

    def is_inside(self, aabb, object_id):
        area_aabb, area_inter = self.compute_intersection(aabb, self.gate_region.inside_mask)
        percent = area_inter / area_aabb
        return percent > 0.5

    def is_outside(self, aabb, object_id):
        area_aabb, area_inter = self.compute_intersection(aabb, self.gate_region.outside_mask)
        percent = area_inter / area_aabb
        return percent > 0.5

    def is_buffer(self, aabb, object_id):
        area_aabb, area_inter = self.compute_intersection(aabb, self.gate_region.buffer_mask)
        percent = area_inter / area_aabb
        return percent > 0.5


class AreaCountingStrategyWriter(AlgoStep):

    def __init__(self, filename, strategy):
        self.logger = get_logger(self)
        self.filename = filename
        self.strategy = strategy

        self.writer = open('{}.csv'.format(self.filename), 'w')
        self.writer.write('in_out,object_id,area_aabb,area_inter\n')

    def process(self, container: AlgoContainer):
        for track in container.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            object_id = track.track_id
            aabb = track.to_tlbr()

            area_aabb, area_inter = self.strategy.compute_intersection(aabb, self.strategy.gate_region.inside_mask)
            self.writer.write('in,{},{},{}\n'.format(object_id, area_aabb, area_inter))

            area_aabb, area_inter = self.strategy.compute_intersection(aabb, self.strategy.gate_region.outside_mask)
            self.writer.write('out,{},{},{}\n'.format(object_id, area_aabb, area_inter))

            area_aabb, area_inter = self.strategy.compute_intersection(aabb, self.strategy.gate_region.buffer_mask)
            self.writer.write('buf,{},{},{}\n'.format(object_id, area_aabb, area_inter))

        return container
