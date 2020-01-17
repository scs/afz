import cv2

from algorithm.counter.gate import GateRegion
from algorithm.object_counter import CountingStrategy


class ObjectCenter(object):
    Top = 'top'
    Center = 'center'
    Bottom = 'bottom'


class PointCountingStrategy(CountingStrategy):

    def __init__(self, object_center, gate_region: GateRegion):
        super().__init__(
            name='point-{}'.format(object_center),
            gate_region=gate_region
        )
        self.object_center = object_center

    def _get_object_point(self, aabb):
        x1, y1, x2, y2 = aabb
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        if self.object_center is ObjectCenter.Top:
            return x, y1
        elif self.object_center is ObjectCenter.Center:
            return x, y
        elif self.object_center is ObjectCenter.Bottom:
            return x, y2
        else:
            raise ValueError('Unknown object center: {}'.format(self.object_center))

    def is_inside(self, aabb, object_id):
        xy = self._get_object_point(aabb)
        return cv2.pointPolygonTest(self.gate_region.inside_contours, xy, False) >= 0

    def is_outside(self, aabb, object_id):
        xy = self._get_object_point(aabb)
        return cv2.pointPolygonTest(self.gate_region.outside_contours, xy, False) >= 0

    def is_buffer(self, aabb, object_id):
        return False


class TopPointCountingStrategy(PointCountingStrategy):
    def __init__(self, gate_region: GateRegion):
        super().__init__(
            object_center=ObjectCenter.Top,
            gate_region=gate_region
        )


class CenterPointCountingStrategy(PointCountingStrategy):
    def __init__(self, gate_region: GateRegion):
        super().__init__(
            object_center=ObjectCenter.Center,
            gate_region=gate_region
        )


class BottomPointCountingStrategy(PointCountingStrategy):
    def __init__(self, gate_region: GateRegion):
        super().__init__(
            object_center=ObjectCenter.Bottom,
            gate_region=gate_region
        )
