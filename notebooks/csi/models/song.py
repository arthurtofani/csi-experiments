import librosa

class Song:
  def __init__(self, path):
    self.feature = None
    self.profiles = None
    self.peaks = None
    self.tokens = None
    self.tmp = {}
    self.path = path
    self.y, self.sr = librosa.load(path)
    self.tempo, self.beats = librosa.beat.beat_track(y=self.y, sr=self.sr, trim=False)

  def samples_per_beat(self):
    return (60 * self.sr) / self.tempo

  def seconds(self):
    return len(self.y) / self.sr
