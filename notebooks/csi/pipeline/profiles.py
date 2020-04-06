import librosa
import numpy as np
from csi.vendor import simplefast
from csi.models.profile import *
from . import kernels
from . import ssm
from csi.util import normalize


class SelfJoinProfile:
  def __init__(self, **kwargs):
    self._kwargs = kwargs

  def execute(self, feature, multiplier, profile_bin):
    results = self._exec(feature, multiplier)
    return Profile(feature.song, results[0], profile_bin, results[1])

  def _exec(self, feature, multiplier):
    raise NotImplementedError


class SimpleFast(SelfJoinProfile):
  def _exec(self, feature, multiplier):
    m = self._kwargs.get('m') or feature.samples_per_beat()
    try:
      s = simplefast.simpleself(feature.data, int(round(m * multiplier)))
    except:
      return (np.array([]), np.array([]))
    return (normalize(s[0]**2), s[1])


class SSMCheckerboardBox(SelfJoinProfile):
  def _exec(self, feature, multiplier):
    L = self._kwargs.get('L') or feature.samples_per_beat()
    _kernel = self._kwargs.get('kernel') or kernels.kernel_checkerboard_box
    s = ssm.recurrence_matrix(feature.data, self=True, **self._kwargs)
    feature.song.tmp['ssm'] = s
    r = ssm.compute_novelty_profile(s, kernel=_kernel(int(round(L * multiplier))), exclude=False)
    #return (r, None)
    return (normalize(r)**2, None)


