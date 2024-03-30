import numpy as np
import cupy
from cupyx.scipy.sparse.linalg import LinearOperator

from pywt import (
    dwt_max_level,
    wavedecn,
    waverecn,
    ravel_coeffs,
    unravel_coeffs,
)

from .utils import to_device_array, from_device_array


class WaveletOperator(LinearOperator):

    def __init__(
        self,
        sz_im,
        wavelet,
        gpu_id,
        stream,
        mode="symmetric"
    ):
        self.sz_im = sz_im
        self.wavelet = wavelet
        self.mode = mode
        self.gpu_id = gpu_id
        self.stream = stream
        self.levels = [dwt_max_level(s, wavelet) for s in sz_im]
        self.min_level = min(self.levels)

        coeffs = wavedecn(
            np.zeros(sz_im, dtype=np.float32),
            self.wavelet,
            mode=self.mode,
            level=self.min_level,
        )
        arr, self.coeff_slices, self.coeff_shapes = ravel_coeffs(coeffs)
        self.shape = (arr.size, np.int64(np.prod(sz_im)))
        self.n_approx_coef = np.sum(self.coeff_shapes[0])

    def _matvec(self, x):

        xp = cupy.get_array_module(x)

        if xp == np:
            h_x = x.reshape(self.sz_im)
        else:
            h_x = from_device_array(
                x, self.gpu_id, self.stream
            ).reshape(self.sz_im)

        coeffs = wavedecn(
            h_x, self.wavelet, mode=self.mode, level=self.min_level
        )

        h_y = ravel_coeffs(coeffs)[0].ravel()[:self.shape[0]]

        if xp == np:
            y = h_y
        else:
            y = to_device_array(h_y, self.gpu_id, self.stream)

        return y

    def _rmatvec(self, x):

        xp = cupy.get_array_module(x)

        if xp == np:
            h_x = x
        else:
            h_x = from_device_array(x, self.gpu_id, self.stream)

        coeffs = unravel_coeffs(
            h_x,
            self.coeff_slices,
            self.coeff_shapes,
            output_format="wavedecn",
        )

        h_y = waverecn(
            coeffs,
            wavelet=self.wavelet,
            mode=self.mode,
        ).ravel()

        if xp == np:
            y = h_y
        else:
            y = to_device_array(h_y, self.gpu_id, self.stream)

        return y


class IdentityOperator(LinearOperator):
    
    def __init__(self, n, gpu_id, stream, dtype='float32'):
        
        self.shape = (n, n)
        self.gpu_id = gpu_id
        self.stream = stream
        self.dtype = np.dtype(dtype)
        
    def _matvec(self, x):
        
        return x.ravel()
    
    def _rmatvec(self, x):
        
        return x.ravel()
