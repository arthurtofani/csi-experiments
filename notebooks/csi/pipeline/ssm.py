import librosa
import numpy as np

def compute_novelty_profile(S, kernel=None, exclude=False):
    N = S.shape[0]
    L = int((kernel[0].size-1)/2)
    M = 2*L + 1
    nov = np.zeros(N)
    S_padded  = np.pad(S,L,mode='constant')
    for n in range(N):
        nov[n] = np.sum(S_padded[n:n+M, n:n+M]  * kernel)
    if exclude:
        right = np.min([L,N])
        left = np.max([0,N-L])
        nov[0:right] = 0
        nov[left:N] = 0
    return nov

def recurrence_matrix(feature, **kwargs):
    args = { 'width':1, 'mode':'affinity', 'sym':True, 'self':True }
    return librosa.segment.recurrence_matrix(feature, args.update(kwargs))

def enhance(ssm, smooth_length, **kwargs):
    return librosa.segment.path_enhance(ssm, smooth_length, kwargs)
