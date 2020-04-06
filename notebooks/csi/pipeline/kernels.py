import numpy as np

def kernel_checkerboard_box(L, **kwargs):
    M = 2*L + 1
    axis = np.arange(-L,L+1)
    kernel = np.outer(np.sign(axis),np.sign(axis))
    return kernel

def kernel_checkerboard_gaussian(L, **kwargs):
    var = kwargs.get('var') or 1
    normalize = kwargs.get('normalize') or True

    taper = np.sqrt(1/2)/(L*var)
    axis = np.arange(-L,L+1)
    gaussian1D = np.exp(-taper**2 * (axis**2))
    gaussian2D = np.outer(gaussian1D,gaussian1D)
    kernel_box = np.outer(np.sign(axis),np.sign(axis))
    kernel = kernel_box * gaussian2D
    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))
    return kernel
