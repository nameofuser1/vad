import numpy as np


def unshared_copy(inlist):
    if isinstance(inlist, list):
        return list(map(unshared_copy, inlist))
    elif isinstance(inlist, np.ndarray):
        return np.array(map(unshared_copy, inlist))

    return inlist


# For you Edward
def scale_features(features):
    scaled_features = features #unshared_copy(features)
    mfcc_buffer = []
    delta1_buffer = []
    delta2_buffer = []

    for file_features in scaled_features:
        for frame_features in file_features:
            mfcc_buffer.append(frame_features[0])
            delta1_buffer.append(frame_features[1])
            delta2_buffer.append(frame_features[2])

    mfcc_mean = np.mean(mfcc_buffer)
    delta1_mean = np.mean(delta1_buffer)
    delta2_mean = np.mean(delta2_buffer)

    mfcc_std = np.std(mfcc_buffer)
    delta1_std = np.std(delta1_buffer)
    delta2_std = np.std(delta2_buffer)

    for file_features in scaled_features:
        for frame_features in file_features:
            for i in range(len(frame_features[0])):
                frame_features[0][i] = (frame_features[0][i] - mfcc_mean) / mfcc_std
                frame_features[1][i] = (frame_features[1][i] - delta1_mean) / delta1_std
                frame_features[2][i] = (frame_features[2][i] - delta2_mean) / delta2_std

    return scaled_features


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]

    return out


