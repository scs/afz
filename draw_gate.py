#!/usr/bin/env python3
import cv2

from algorithm.counter.gate import Stairs_960x540

cv2.imwrite('stairs-inside.jpg', 255 * Stairs_960x540.inside_mask)
cv2.imwrite('stairs-outside.jpg', 255 * Stairs_960x540.outside_mask)
cv2.imwrite('stairs-buffer.jpg', 255 * Stairs_960x540.buffer_mask)
