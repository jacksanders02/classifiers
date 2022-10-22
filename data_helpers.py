from pathlib import Path
import numpy as np
import time


# Load a data file (if file is .npy, load, else load and then convert to .npy)
def loadfile(file: str, extension: str = ""):
    start = time.time()
    if Path(f'data/{file}{extension}.npy').is_file():
        ext = 'npy'
        x = np.load(f'data/{file}{extension}.npy')
    else:
        ext = 'txt'
        x = np.loadtxt(open(f'data/{file}{extension}.txt'), delimiter=",")
        np.save(f'data/{file}{extension}.npy', x)
        print(f'Converted dataset {file}{extension}.txt to npy file for faster loading next time this dataset is used.')

    print(f'Loaded file {file}{extension}.{ext} in {round(time.time() - start, 2)} seconds. (To 2 D.P.)')

    return x


# split data into training and testing sets
def split_data(xs: np.array, leave_out: int = None):
    if leave_out is not None:
        return np.delete(xs, leave_out, axis=0), xs[leave_out]
    else:
        return xs[0::2], xs[1::2]