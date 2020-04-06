import librosa

class Song:
  def __init__(self, path):
    self.feature = None
    self.profiles = None
    self.peaks = None
    self.tokens = None
    self.path = path
    self.y, self.sr = librosa.load(path)
    self.tempo, self.beats = librosa.beat.beat_track(y=self.y, sr=self.sr, trim=False)
