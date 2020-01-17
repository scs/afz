import os


def get_base_path():
    home = os.path.expanduser('~')
    return os.path.join(home, 'sbb-afz')


def get_output_path(filename):
    return os.path.join(get_base_path(), 'output', filename)


def get_pickle_path(filename, fps, tag):
    return os.path.join(get_base_path(), 'output', 'algo-{}-{}fps-{}'.format(filename, fps, tag))


def get_video_input_path(filename):
    return os.path.join(get_base_path(), 'videos', '{}.mp4'.format(filename))


def get_label_input_path(filename):
    return os.path.join(get_base_path(), 'labels', '{}.csv'.format(filename))
