from multiprocessing import Process, Queue, Event
import pyaudio


class Recoder:

    def __init__(self, frame_rate, period):
        #
        #   Period in ms
        #
        self.proc = None
        self.running = Event()
        self.samples_queue = Queue()
        self.frame_rate = frame_rate
        self.chunk_size = (frame_rate*period) / 1000
        self.channels = 1

        print(self.chunk_size)

        self._pa = pyaudio.PyAudio()
        self._stream = None

    def start(self):
        if self.proc is None:
            self._stream = self._pa.open(format=pyaudio.paInt8,
                                         channels=self.channels,
                                         rate=self.frame_rate,
                                         input=True,
                                         frames_per_buffer=self.chunk_size)

            self.running.set()
            self.proc = Process(target=self._recording_loop, args=[self.samples_queue, self.running, self._stream,
                                                                   self.chunk_size])
            self.proc.start()

    def stop(self):
        if self.proc is not None:
            self.running.clear()
            self.proc.join()

        self._stream.close()
        self._pa.terminate()

    def empty(self):
        return self.samples_queue.empty()

    def read(self):
        res = []
        while not self.samples_queue.empty():
            res.append(self.samples_queue.get())

        return res

    def _recording_loop(self, samples_queue, running, stream, chunk_size):
        stream.start_stream()

        while running.is_set():
            samples_queue.put(stream.read(chunk_size))

        stream.stop_stream()