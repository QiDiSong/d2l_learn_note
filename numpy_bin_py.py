import numpy as np

data = np.random.uniform(0, 1, size=[1, 544, 960, 4]).astype(np.uint8)
data.tofile('input.bin')
dat = np.fromfile('input.bin')

dat = np.fromfile('input.bin', dtype=np.uint8)