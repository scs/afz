import os
import pickle

from algorithm.algorithm import AlgoStep, AlgoContainer


class PickleWriter(AlgoStep):

    def __init__(self, pickle_path):
        self.pickle_path = pickle_path
        if not os.path.exists(self.pickle_path):
            os.mkdir(self.pickle_path)

    def process(self, container: AlgoContainer):
        with open(os.path.join(self.pickle_path, '{}.pickle'.format(container.position)), 'wb') as f:
            pickle.dump(container.without_frame(), f)

        return container


class PickleLoader(AlgoStep):

    def __init__(self, pickle_path):
        self.pickle_path = pickle_path
        if not os.path.exists(self.pickle_path):
            os.mkdir(self.pickle_path)

    def process(self, container: AlgoContainer):
        with open(os.path.join(self.pickle_path, '{}.pickle'.format(container.position)), 'rb') as f:
            container_p = pickle.load(f)

        return container_p.extend_with(frame=container.frame)
