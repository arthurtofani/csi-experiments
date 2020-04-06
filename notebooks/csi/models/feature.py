class Feature:
  def __init__(self, song, data, name=None):
    self.song = song
    self.data = data
    self.name = name

  def sample_rate(self):
    return (self.song.sr  * self.data.size) / self.song.y.size

  def samples_per_beat(self):
    return (60 * self.sample_rate()) / self.song.tempo
