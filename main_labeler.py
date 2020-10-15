#!/usr/bin/env python3
import os

import cv2


def label_video(video_dir, video_filename, label_file):
    video_path = os.path.join(video_dir, video_filename)

    print('Opening {}...'.format(video_path))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Error')

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print('frame_width = {}'.format(frame_width))
    print('frame_height = {}'.format(frame_height))
    print('fps = {}'.format(fps))

    label_file.write('frame,timestamp,entered,left\n')

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        pos_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        """
        if pos_frames % 100 == 0:
            frame_path = os.path.join(video_dir, 'frame-{:05}.jpg'.format(pos_frames))
            cv2.imwrite(frame_path, frame)
        """

        cv2.imshow('frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        has_entered = 1 if key == ord('e') else 0
        has_left = 1 if key == ord('l') else 0

        label_file.write('{},{},{},{}\n'.format(pos_frames, pos_ms, has_entered, has_left))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    label_filename = 'gate1.csv'
    video_dir = 'new_videos/'
    video_filename = 'gate1.mp4'
    with open(label_filename, 'w') as f:
        label_video(video_dir, video_filename, f)
