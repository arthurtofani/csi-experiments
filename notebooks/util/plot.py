import librosa
import librosa.display as display
import matplotlib.pyplot as plt

def plot(dt, sr=22050):
    plt.figure(figsize=(12, 5))
    onset_frames = librosa.onset.onset_detect(onset_envelope=dt, sr=sr)
    plt.plot(dt)
    plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
    plt.tight_layout()


def plot_chroma(chroma):
    plt.figure(figsize=(14, 6))
    plt.subplot(2,1,1)
    display.specshow(chroma, y_axis='chroma', x_axis='frames')
    plt.title('chroma')
    plt.colorbar()
    plt.tight_layout()


def plot_ssm(dt):
    rec = librosa.segment.recurrence_matrix(dt, mode='affinity', self=True)
    data = librosa.segment.path_enhance(rec, 5, window='hann', n_filters=1)
    plt.figure(figsize=(8, 8))
    display.specshow(data, x_axis='frames')
    plt.title('SSM')
    plt.tight_layout()
