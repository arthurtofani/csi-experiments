import librosa
import numpy as np
from csi.models import feature as model

class Feature:
  def __init__(self, **kwargs):
    self._kwargs = kwargs

  def execute(self, song):
    data = self._exec(song)
    return model.Feature(song, data)

  def _exec(self, song):
    raise NotImplementedError

  def _signal_params(self, song):
    h = {'y': song.y, 'sr': song.sr}
    h.update(self._kwargs)
    return h


class ChromagramCQT(Feature):
  def _exec(self, song):
    return librosa.feature.chroma_cqt(**self._signal_params(song))


class ChromagramCENS(Feature):
  def _exec(self, song):
    return librosa.feature.chroma_cens(**self._signal_params(song))

# import code; code.interact(local=dict(globals(), **locals()))

#class CQT(Feature):
#  BINS_PER_OCTAVE=36
#  N_OCTAVES=7
#
#  def _exec(self, song):
#    return generate_feature(self, song)
#
#  def generate_feature(self, song):
#    params = { 'bins_per_octave': BINS_PER_OCTAVE,
#               'n_bins': self.N_OCTAVES * self.BINS_PER_OCTAVE }
#    return librosa.core.cqt(y=song.y, sr=song.sr, params.update(self._kwargs))
#
#
#class BeatSyncCQT(self):
#  def _exec(self, song):
#    cqt = BeatSyncCQT(self._kwargs).generate_feature(song)
#    C = librosa.amplitude_to_db(cqt, ref=np.max)
#    return librosa.util.sync(C, song.beats, aggregate=np.median)
