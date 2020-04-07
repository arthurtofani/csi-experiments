import librosa
import librosa.display as display
import matplotlib.pyplot as plt
import numpy as np

PLOT_COLORS = ['b','g','r', 'c', 'm', 'y', 'pink', 'gray', 'orange', 'brown', 'purple', 'black' ]

def samples_per_beat(bpm, sr):
  return (60 / bpm) * sr

def normalize(x):
  arr = np.array(x) - min(x)
  return (arr/np.amax(arr))**2

def bit_quantize(real_value, n_bits):
  if real_value > 1.0 or real_value < 0:
    raise ValueError("Must be a value between 0 and 1")
  return int(2 ** n_bits * real_value)

def plot_signal(song):
  plt.plot(song.y)
  plt.text(-3, -3, 'frames/second = %s' % song.sr)
  plt.text(-3, -4.5, 'frames/beat = %s' % song.samples_per_beat())
  plt.title(song.path)


def plot_peaks(peaks, profiles, **kwargs):
  for i in range(len(peaks)):
    peak_list = peaks[i]
    peak_idx = [profiles[i].data[j] for j in peak_list]
    args = {'color': PLOT_COLORS[i]}
    args.update(kwargs)
    plt.plot(peak_list, peak_idx, 'o', **args)
  #plt.colorbar()

def plot_chroma(song, **kwargs):
  chroma = song.feature.data
  args = {'y_axis':'chroma', 'x_axis':'frames'}
  args.update(kwargs)
  display.specshow(chroma, **args)
  plt.text(-3, -3, 'frames/second = %s' % song.feature.sample_rate())
  plt.text(-3, -4.5, 'frames/beat = %s' % song.feature.samples_per_beat())
  plt.title(song.path)
  #plt.colorbar()

def plot_constellation(peaks):
  for i in range(len(peaks)):
    peak_list = peaks[i]
    plt.plot(peak_list, (peak_list/peak_list)*i, 'o', color=PLOT_COLORS[i])
  #plt.colorbar()

def plot_ssm(ssm):
  display.specshow(ssm, x_axis='frames')
  plt.title('SSM')

def plot_profiles(profiles, **kwargs):
  for i in range(len(profiles)):
    args = {'color': PLOT_COLORS[i]}
    args.update(kwargs)
    plt.plot(profiles[i].data, **args)
  #plt.colorbar()
