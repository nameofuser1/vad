from multiprocessing import Process, Queue, Event
import alsaaudio


class Recoder:

    def __init__(self, frame_rate, period):

        self.proc = None
        self.running = Event()
        self.samples_queue = Queue()
        self.frame_rate = frame_rate
        self.period = period

    def start(self):
        if self.proc is None:
            self.running.set()
            self.proc = Process(target=self._recording_loop, args=[self.samples_queue, self.running, self.frame_rate,
                                                                   self.period])
            self.proc.start()

    def stop(self):
        if self.proc is not None:
            self.running.clear()
            self.proc.join()

    def empty(self):
        return self.samples_queue.empty()

    def read(self):
        res = []
        while not self.samples_queue.empty():
            res.append(self.samples_queue.get())

        return res

    def _recording_loop(self, samples_queue, running, frame_rate, period):

        rec = alsaaudio.PCM(type=alsaaudio.PCM_CAPTURE, mode=alsaaudio.PCM_NORMAL)
        rec.setchannels(1)
        rec.setrate(frame_rate)
        rec.setperiodsize(period)
        rec.setformat(alsaaudio.PCM_FORMAT_S8)

        while running.is_set():
            samples_queue.put(rec.read())

        rec.close()
