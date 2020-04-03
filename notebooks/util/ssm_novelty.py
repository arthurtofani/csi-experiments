import librosa
import numpy as np


def compute_kernel_checkerboard_box(L):
    M = 2*L + 1
    axis = np.arange(-L,L+1)
    kernel = np.outer(np.sign(axis),np.sign(axis))
    return kernel

def compute_kernel_checkerboard_gaussian(L, var=1, normalize=True):
    taper = np.sqrt(1/2)/(L*var)
    axis = np.arange(-L,L+1)
    gaussian1D = np.exp(-taper**2 * (axis**2))
    gaussian2D = np.outer(gaussian1D,gaussian1D)
    kernel_box = np.outer(np.sign(axis),np.sign(axis))
    kernel = kernel_box * gaussian2D
    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))
    return kernel

def compute_novelty_SSM(S, kernel=None, exclude=False):
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

def detect_novelty(S, L=80, var=0.2):
    kernel = compute_kernel_checkerboard_gaussian(L=L, var=var)
    rec = librosa.segment.recurrence_matrix(S, mode='affinity', self=True)
    #data = librosa.segment.path_enhance(rec, 5, window='hann', n_filters=1)
    return compute_novelty_SSM(rec, kernel, L)

def normalize_rows(x):
    return (x/np.amax(x))**2
