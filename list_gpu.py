#!/usr/bin/env python3

import tensorflow as tf
from keras import backend as K

if __name__ == '__main__':
    K.tensorflow_backend._get_available_gpus()
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 4})
    sess = tf.Session(config=config)
    K.set_session(sess)

    from tensorflow.python.client import device_lib

    print('>>>>>>>>>>>>>>> local_devices >>>>>>>>>>>>>>>>>>>')
    print(device_lib.list_local_devices())
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
