from . import features
from . import kernels
from . import peak_detection
from . import profiles
from . import ssm
from . import tokenizers
from csi.models.song import *
import numpy as np

class Pipeline:
  _feature_strategy = None
  _profile_strategy = None
  _profile_multipliers = None
  _profile_bins = None
  _peaks_strategy = None
  _tokenizer_strategy = None

  songs = None
  files = None

  def run(self, files):
    self.files = files
    self._read_files()
    if self._feature_strategy is None: return
    self._process_feature()
    if self._profile_strategy is None: return
    self._create_profiles()
    if self._peaks_strategy is None: return
    self._detect_peaks()
    if self._tokenizer_strategy is None: return
    self._tokenize()
    print("Done")

  def use_feature(self, feature_strategy):
    self._feature_strategy = feature_strategy
    return self

  def create_profiles(self, profile_strategy, multipliers, bins=None):
    self._profile_strategy = profile_strategy
    self._profile_multipliers = multipliers
    self._profile_bins = (bins or (np.arange(len(multipliers))+1))
    return self

  def detect_peaks(self, peaks_strategy, **kwargs):
    self._peaks_strategy = peaks_strategy
    return self

  def tokenize(self, tokenizer, **kwargs):
    self._tokenizer_strategy = tokenizer
    return self

  def _read_files(self):
    print("Reading files")
    self.songs = []
    for file in self.files:
      print('reading file %s ' % file)
      self.songs.append(Song(file))

  def _process_feature(self):
    print("Processing features")
    _features = []
    for song in self.songs:
      print('processing feature => %s' % song.path)
      song.feature = self._feature_strategy.execute(song)

  def _create_profiles(self):
    print("Creating profiles")
    for song in self.songs:
      song.profiles = []
      l = len(self._profile_multipliers)
      for i in range(l):
        print('creating profiles - %s of %s ' % (i+1, l))
        multiplier = self._profile_multipliers[i]
        profile_bin = self._profile_bins[i]
        prfl = self._profile_strategy.execute(song.feature, multiplier, profile_bin)
        song.profiles.append(prfl)

  def _detect_peaks(self):
    print("Detecting peaks")
    for song in self.songs:
      song.peaks = []
      l = len(song.profiles)
      i = -1
      for profile in song.profiles:
        peaks = self._peaks_strategy.execute(profile)
        song.peaks.append(peaks)

  def _tokenize(self):
    print("Creating tokens")
    for song in self.songs:
      song.tokens = []
      song.tokens = self._tokenizer_strategy.execute(song)
