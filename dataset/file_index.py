import h5py
import numpy as np

__all__ = ['save_file_index', 'load_file_index']


def create_file_index(path, offset=1):
    index = []
    line_number = 0
    seek_offset = 0

    with open(path, 'rb') as f:
        for line in f:
            if line_number < offset:
                line_number += 1
                seek_offset += len(line)
                continue

            index.append(seek_offset)
            seek_offset += len(line)
            line_number += 1

    return np.asarray(index, dtype=np.uint32)


def save_file_index(path, offset=1):
    index_path = get_index_path(path)
    index_name = get_index_name(path)
    print(index_name)

    with h5py.File(index_path, 'w') as f:
        f.create_dataset(index_name, data=create_file_index(path, offset))
        f.close()


def load_file_index(path):
    with h5py.File(get_index_path(path), 'r') as f:
        index = f[get_index_name(path)][:]
        f.close()
        return index


def get_index_path(path):
    return path.split('.')[0] + '.index'


def get_index_name(path):
    return (path.split('/')[-1]).split('.')[0] + '.index'
