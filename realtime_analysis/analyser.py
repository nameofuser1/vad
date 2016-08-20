import abc


class Analyser:

    def __init__(self):
        pass

    @abc.abstractmethod
    def load_init_inactive_frames(self, frames):
        return

    @abc.abstractmethod
    def feed_frame(self, frame):
        return