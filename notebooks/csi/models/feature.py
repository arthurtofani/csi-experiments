class Feature:
  def __init__(self, song, data, name=None):
    self.song = song
    self.data = data
    self.name = name

  def sample_rate(self):
    return len(self.song.feature.data[0])/self.song.seconds()

  def samples_per_beat(self):
    return (60 * self.sample_rate()) / self.song.tempo
