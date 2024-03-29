import logging
import numpy as np
import cupy

from .thresholding.thresholding import soft_thresholding
from .utils import to_device_array, from_device_array


logger = logging.getLogger(__name__)

n_devices = cupy.cuda.runtime.getDeviceCount()
mempools = [[] for _ in range(n_devices)]
pinned_mempools = [[] for _ in range(n_devices)]
for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        mempools[gpu_id] = cupy.cuda.MemoryPool()
        cupy.cuda.set_allocator(mempools[gpu_id].malloc)
        pinned_mempools[gpu_id] = cupy.cuda.PinnedMemoryPool()
        cupy.cuda.set_pinned_memory_allocator(pinned_mempools[gpu_id].malloc)


def clean_up_gpu(gpu_id):
    with cupy.cuda.Device(gpu_id):
        mempools[gpu_id].free_all_blocks()
        pinned_mempools[gpu_id].free_all_blocks()


class WaveletFISTA:

    def __init__(self, model, wavelet_op):

        self.model = model
        self.gpu_id = model.gpu_id
        self.stream = model.stream
        self.W = wavelet_op

    def run(self, data, niter, reg, lipschitz_const, init=None):

        f = to_device_array(
            data,
            ravel=True,
            gpu_id=self.gpu_id,
            stream=self.stream,
            dtype_real='float32',
            dtype_complex='complex64'
        )

        if init is None:
            with cupy.cuda.Device(self.gpu_id), self.stream as _:
                y = cupy.zeros((self.model.shape[1],), dtype=self.model.dtype)
        else:
            y = to_device_array(
                init,
                ravel=True,
                gpu_id=self.gpu_id,
                stream=self.stream,
                dtype_real='float32',
                dtype_complex='complex64'
            )

        with cupy.cuda.Device(self.gpu_id), self.stream as _:
            x_old = y.copy()

        t_old = 1.0

        for _ in range(niter):
            with cupy.cuda.Device(self.gpu_id), self.stream as _:
                proj = self.model.matvec(y)
                res = proj - f
                backproj = self.model.rmatvec(res)

            with cupy.cuda.Device(self.gpu_id), self.stream as stream:
                x_new = y - backproj / lipschitz_const
                coeffs = self.W.matvec(x_new[:self.W.shape[1]])
                coeffs = soft_thresholding(
                    coeffs, reg, gpu_id=self.gpu_id, stream=stream
                )
                x_new[:self.W.shape[1]] = self.W.rmatvec(coeffs)

            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_old * t_old)) / 2.0
            wgt_new = 1.0 + (t_old - 1.0) / t_new
            wgt_old = (1.0 - t_old) / t_new

            with cupy.cuda.Device(self.gpu_id), self.stream as stream:
                y = wgt_new * x_new + wgt_old * x_old
                x_old, x_new = x_new, x_old

            t_old, t_new = t_new, t_old

        if cupy.get_array_module(data) == np:
            return from_device_array(x_old, self.gpu_id, self.stream)
        else:
            return x_old
