#!/usr/bin/env python3
import cv2

from algorithm.counter.gate import Gate1, Gate2, Hallway, Stairs, Stairs_960x540

cv2.imwrite('stairs-inside.jpg', 255 * Stairs_960x540.inside_mask)
cv2.imwrite('stairs-outside.jpg', 255 * Stairs_960x540.outside_mask)
cv2.imwrite('stairs-buffer.jpg', 255 * Stairs_960x540.buffer_mask)

if False:
    cv2.imwrite('gate1-inside.jpg', 255 * Gate1.inside_mask)
    cv2.imwrite('gate1-outside.jpg', 255 * Gate1.outside_mask)

    cv2.imwrite('gate2-inside.jpg', 255 * Gate2.inside_mask)
    cv2.imwrite('gate2-outside.jpg', 255 * Gate2.outside_mask)

    cv2.imwrite('hallway-inside.jpg', 255 * Hallway.inside_mask)
    cv2.imwrite('hallway-outside.jpg', 255 * Hallway.outside_mask)

    cv2.imwrite('stairs-inside.jpg', 255 * Stairs.inside_mask)
    cv2.imwrite('stairs-outside.jpg', 255 * Stairs.outside_mask)
