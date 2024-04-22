import numpy as np


def scaling(datx):
    ndatx = []
    for dx in datx:
        mean, std = np.average(dx, axis=0), np.std(dx, axis=0)
        ndatx.append((dx-mean)/std)
    return np.array(ndatx)