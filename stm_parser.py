from functools import partial
import numpy as np

def parse(fname):
	starts = np.array([], dtype=np.float32)
	ends = np.array([], dtype=np.float32)

	with open(fname, 'rb') as f:
		for chunk in iter(partial(f.readline, 1024), ""):
			items = chunk.split(' ')
			if (len(items) < 7) or (items[6].strip() == "ignore_time_segment_in_scoring"):
				continue

			starts = np.append(starts, float(items[3]))
			ends = np.append(ends,  float(items[4]))

	return [starts, ends]


def get_samples_indices(fname, samplerate=16384):
	starts, ends = parse(fname)
	starts = (starts*samplerate).astype(np.int32, copy=False)
	ends = (ends*samplerate).astype(np.int32, copy=False)

	return starts, ends


print(get_samples_indices("DanBarber_2010.stm"))
