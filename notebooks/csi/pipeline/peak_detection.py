import librosa
import scipy

class PeakDetection:
  def __init__(self, **kwargs):
    self._kwargs = kwargs

  def execute(self, profile):
    return self._exec(profile)

class SpectralFlux(PeakDetection):
  def _exec(self, profile):
    params = {'y': profile.data, 'sr': int(profile.sample_rate())}
    params.update(self._kwargs)
    return librosa.onset.onset_detect(**params)

class FindPeaks(PeakDetection):
  def _exec(self, profile):
    params = {'height': 0.08 , 'distance': 10 }
    params.update(self._kwargs)
    return scipy.signal.find_peaks(profile.data, **params)[0]
