import librosa
import librosa.display as display
import matplotlib.pyplot as plt
import numpy as np

def samples_per_beat(bpm, sr):
  return (60 / bpm) * sr

def normalize(x):
  arr = np.array(x) - min(x)
  return (arr/np.amax(arr))**2

def bit_quantize(real_value, n_bits):
  if real_value > 1.0 or real_value < 0:
    raise ValueError("Must be a value between 0 and 1")
  return int(2 ** n_bits * real_value)

def plot_signal(signal, sr=22050):
  plt.figure(figsize=(12, 5))
  onset_frames = librosa.onset.onset_detect(onset_envelope=dt, sr=sr)
  plt.plot(dt)
  plt.tight_layout()

def plot_onsets(onsets):
  plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')

def plot_chroma(chroma):
  plt.figure(figsize=(14, 6))
  plt.subplot(2,1,1)
  display.specshow(chroma, y_axis='chroma', x_axis='frames')
  plt.title('chroma')
  plt.colorbar()
  plt.tight_layout()


def plot_ssm(ssm):
  plt.figure(figsize=(8, 8))
  display.specshow(ssm, x_axis='frames')
  plt.title('SSM')
  plt.tight_layout()
