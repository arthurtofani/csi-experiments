import matplotlib.pyplot as plt
import numpy as np
import scipy
from csi.pipeline import kernels
import librosa
from librosa import display
from csi.vendor import simplefast
from csi.pipeline import ssm as ssmlib
from collections import defaultdict

def normalize(x):
    arr = np.array(x)
    arr2 = arr - arr.min()
    return (arr2/np.amax(arr2))

class Audio:
  def __init__(self, audio_file):
    self.filename = audio_file
    self.y, self.sr = librosa.load(audio_file)
    self.tempo, self.beats = librosa.beat.beat_track(y=self.y, sr=self.sr, trim=False)
    self.seconds = len(self.y)/self.sr
    self.samples_per_beat = (60 * self.sr) / self.tempo

  def extract_chroma(self, hop_length=2**11):
    self.chroma = librosa.feature.chroma_cens(self.y, self.sr, hop_length=hop_length)
    feature_sr = len(self.chroma[0])/self.seconds
    self.chroma_spb = (60 * feature_sr) / self.tempo
    self.chroma_length = len(self.chroma[0])

  def extract_cic(self, hop_length=2**11):
    self.extract_chroma(hop_length=hop_length)
    self.chroma = self.calc_cic(self.chroma)
    feature_sr = len(self.chroma[0])/self.seconds
    self.chroma_spb = (60 * feature_sr) / self.tempo
    self.chroma_length = len(self.chroma[0])


  def extract_ssm(self):
    ssm = librosa.segment.recurrence_matrix(self.chroma, width=1, mode='affinity', sym=True, self=True)
    self.ssm = librosa.segment.path_enhance(ssm, 20, window='hann', n_filters=5)

  # https://github.com/rcaborges/chroma-interval-content/blob/master/main.py
  def calc_cic(self, crm_arr):
    crmip = np.zeros(12)
    tmtx = []
    for crmi in crm_arr.T:
        row_d = []
        for d in np.arange(-5,7,1):
            sum_crm = 0
            for i in range(12):
                sum_crm = sum_crm + (crmip[i]*crmi[(i+d)%12])
            row_d.append(sum_crm)
        #row_d = row_d/np.linalg.norm(row_d)
        row_d = (row_d-np.min(row_d))/(np.max(row_d)-np.min(row_d))
        crmip = crmi
        if not np.isnan(row_d).any():
            tmtx.append(row_d)
    tmtx = np.array(tmtx)
    tmtx = tmtx[1:-1,:].T
    tmtx = tmtx.tolist()
    return np.array(tmtx)

  def extract_segments(self, size=128, max_levels=6):
    k_size = int(size * self.chroma_spb)
    self.segments_kernel_size = size
    self.segments_max_levels = max_levels
    self.segments = ProfileSegment(self.ssm, 0, len(self.ssm[0]), k_size, max_levels=max_levels)

  def list_peaks(self):
    return self.segments.list_peaks()

  def plot_segments(self):
    display.specshow(self.ssm, **{'y_axis':'frames', 'x_axis':'frames'})
    plt.plot(self.segments_kernel_size, color='white', alpha=0.9)
    self.segments.plot()
    plt.show()

  def plot_peaks(self):
    pks = self.list_peaks()
    ssm = self.ssm
    display.specshow(self.ssm, **{'y_axis':'frames', 'x_axis':'frames', 'alpha':0.5})
    bin_size = len(self.ssm[0])/len(pks)
    for idx in range(len(pks)):
      #plt.vlines(pks[idx], idx*bin_size, idx*bin_size+bin_size, color='white', alpha=0.8, linestyle='-', label='Onsets')
      xaxis = np.zeros(len(pks)) + idx*bin_size
      plt.plot(pks[idx], pks[idx]/pks[idx] + idx*bin_size + 20  , 'o')
    #plt.plot(k_size, color='white', alpha=0.9)
    #plt.plot(profile * len(ssm[0]), color='yellow', alpha=0.4)


  def plot_profile(self, multiplier=32):
    m_size = int(multiplier * self.chroma_spb)
    kernel = kernels.kernel_checkerboard_gaussian(m_size)
    chroma_length = len(self.chroma[0])
    profile = normalize(ssmlib.compute_novelty_profile(self.ssm, kernel))
    peaks = scipy.signal.find_peaks(profile, **{'height': 0.2 , 'distance': m_size })[0]
    m_line = np.zeros(m_size) + chroma_length/2
    display.specshow(self.ssm, **{'y_axis':'frames', 'x_axis':'frames'})
    plt.plot(m_line, color='white', alpha=0.9)
    plt.plot(profile * len(self.ssm[0]), color='yellow', alpha=0.4)
    plt.vlines(peaks, 0, chroma_length, color='gray', alpha=0.4, linestyle='--', label='Onsets')
    plt.hlines(peaks, 0, chroma_length, color='gray', alpha=0.4, linestyle='--', label='Onsets')

  def plot_chroma(self):
    display.specshow(self.chroma, **{'y_axis':'chroma', 'x_axis':'frames'})

  def plot_ssm(self):
    display.specshow(self.ssm, **{'y_axis':'frames', 'x_axis':'frames'})


class ProfileSegment:
    peaks = None
    def __init__(self, ssm, pos_ini, pos_end, k_size, max_levels=999, uid='0', level=1):
        if k_size < 2 or max_levels==0:
            return
        if (pos_end - pos_ini) < (k_size * 2):
            return
        self.uid = uid
        self.level = level
        self.max_levels = max_levels
        self.pos_ini = int(pos_ini)
        self.pos_end = int(pos_end)
        self.ssm = ssm
        self.k_size = k_size
        self.segments = []
        self.compute()

    def list_peaks(self):
        l = self._get_peaks_rec(defaultdict(list))
        arr = []
        for key in np.sort([int(k) for k in l.keys()]):
            elements = np.unique(l[str(key)])
            arr.append(elements)
        return arr

    def get_points(self):
        return np.sort(np.append(self.peaks, [self.pos_ini, self.pos_end]))


    def ssm_block(self):
        return self.ssm[self.pos_ini:(self.pos_end -1)]

    def compute(self):
        self.compute_peaks()
        self.compute_segments()

    def plot(self):
        plt.vlines(self.peaks, self.pos_ini, self.pos_end, color='white', alpha=0.4, linestyle='--', label='Onsets')
        plt.hlines(self.peaks, self.pos_ini, self.pos_end, color='white', alpha=0.4, linestyle='--', label='Onsets')
        for s in self.segments:
            s.plot()

    def compute_peaks(self):
        kernel = kernels.kernel_checkerboard_gaussian(int(self.k_size))
        profile = normalize(ssmlib.compute_novelty_profile(self.ssm_block(), kernel))
        self.peaks = scipy.signal.find_peaks(profile, **{'height': 0.05 , 'distance': self.k_size })[0]
        self.peaks += self.pos_ini

    def compute_segments(self):
        points = self.get_points()
        idx=-1
        self.segments = []
        for i in zip(points, points[1:]):
            idx+=1
            pos_ini, pos_end = i
            segment = ProfileSegment(self.ssm, pos_ini, pos_end, int(self.k_size/2), self.max_levels -1, self.uid + '.' + str(idx), self.level+1)
            if segment.peaks is not None:
                self.segments.append(segment)

    def _get_peaks_rec(self, peak_dict):
        for p in self.get_points():
            peak_dict[str(self.level)].append(p)
        for s in self.segments:
            s._get_peaks_rec(peak_dict)
        if self.level==1:
            return dict(peak_dict)
