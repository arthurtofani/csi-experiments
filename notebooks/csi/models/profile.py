class Profile:
  def __init__(self, song, data, profile_bin, indices=[]):
    self.data = data
    self.indices = indices
    self.bin = profile_bin
    self.song = song

  def sample_rate(self):
    return (self.song.sr  * self.data.size) / self.song.y.size

  def samples_per_beat(self):
    return (60 * self.sample_rate()) / self.song.tempo
