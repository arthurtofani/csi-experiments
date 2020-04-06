import librosa
import numpy as np
import scipy
import matplotlib.pyplot as plt
from . import ssm as ssmlib
from . import util


class Profile:
    tempo = None
    pace = None
    ssm = None
    multipliers = None
    profiles = None
    peaks = None

    def __init__(self, song, ssm, multipliers):
        self.ssm = ssm
        self.multipliers = multipliers
        self.tempo = util.samples_per_beat(song.tempo, song.sr)/(song.y.size/ssm[0].size)
        self.pace = int(self.tempo * 8)

    def compute(self):
        self.profiles = []
        self.peaks = []
        for m in self.multipliers:
            kernel = ssmlib.compute_kernel_checkerboard_box(L=self.pace*m)
            r = ssmlib.compute_novelty_SSM(self.ssm, kernel=kernel)
            pks = scipy.signal.find_peaks(r, height=0.08, distance=10)[0]
            self.profiles.append(r)
            self.peaks.append(pks)
        return self.profiles

    def plot_profiles(self):
        plt.figure(figsize=(18, 4))
        for m in np.arange(len(self.multipliers)):
            plt.plot(ssmlib.normalize_rows(self.profiles[m]))
        plt.show()

    def plot_peaks(self):
        plt.figure(figsize=(18, 4))
        indexes = np.arange(len(self.multipliers))
        for m in indexes:
            plt.plot(self.peaks[m], np.zeros(len(self.peaks[m]))+m, 'o')
        plt.show()



