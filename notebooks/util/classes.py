import librosa
import numpy as np
import scipy
import matplotlib.pyplot as plt
from . import ssm_novelty as ssmlib
from . import util

class Song:
    BINS_PER_OCTAVE = 12 * 3
    N_OCTAVES = 7

    y = None
    sr = None
    path = None
    _chroma_cens = None
    _chroma_cqt = None
    _beat_sync_cqt = None
    tempo = None
    beats = None
    profile = None

    def __init__(self, path):
        self.path = path
        self.y, self.sr = librosa.load(path)
        self.tempo, self.beats = librosa.beat.beat_track(y=self.y, sr=self.sr, trim=False)

    def chroma_cens(self, **kwargs):
        if self._chroma_cens is None:
            self._chroma_cens = librosa.feature.chroma_cens(y=self.y, sr=self.sr, **kwargs)
        return self._chroma_cens

    def chroma_cqt(self, **kwargs):
        if self._chroma_cqt is None:
            self._chroma_cqt = librosa.feature.chroma_cqt(y=self.y, sr=self.sr, **kwargs)
        return self._chroma_cqt

    def beat_sync_cqt(self):
        if self._beat_sync_cqt is None:
            cqt = librosa.core.cqt(y=self.y, sr=self.sr,
                                   bins_per_octave=self.BINS_PER_OCTAVE,
                                   n_bins=self.N_OCTAVES * self.BINS_PER_OCTAVE)
            C = librosa.amplitude_to_db(cqt, ref=np.max)
            self._beat_sync_cqt = librosa.util.sync(C, self.beats, aggregate=np.median)
        return self._beat_sync_cqt

    def create_profile(self, ssm, multipliers=[4, 8]):
        pfl = Profile(self, ssm, multipliers)
        r = pfl.compute()
        self.profile = pfl
        return r




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



