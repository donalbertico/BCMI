import mne
import numpy as np


raw = mne.io.read_raw_bdf("data/e01.bdf")
raw.plot(block=True)
