import numpy as np
import cupy
import logging


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


def estimate_lipschitz_constant(model, niter, h_x0=None, seed=0):
    """Returns Lipschitz constant using Power iteration.

    Args:
        model (cupyx.scipy.sparse.linalg.LinearOperator): model
        niter (int): number of Power iterations
        seed (float): seed for random number generator

    Returns:
        float: Lipschitz constant
    """

    n = model.shape[1]
    gpu_id = model.gpu_id
    stream = model.stream

    state = np.random.get_state()  # save the current state of RGN
    np.random.seed(seed)

    # initialization with uniform random generator

    if h_x0 is None:
        if np.iscomplexobj(np.ones((1,), dtype=np.dtype(model.dtype))):
            h_x0 = (
                np.random.uniform(size=(n,))
                + np.array([1j]) * np.random.uniform(size=(n,))
            )
        else:
            h_x0 = np.random.uniform(size=(n,))

    np.random.set_state(state)  # restore the original state

    with cupy.cuda.Device(gpu_id), stream as _:
        x0 = cupy.empty((n,), dtype=model.dtype)
        x0.set(h_x0.astype(x0.dtype).ravel(), stream=stream)
        stream.synchronize()
        x0 /= cupy.linalg.norm(x0)

    # power iterations

    for i in range(niter):
        x1 = model.rmatvec(model.matvec(x0))
        with cupy.cuda.Device(gpu_id), stream as _:
            x0 = x1 / cupy.linalg.norm(x1)

    # estimate lipschitz_constant

    x1 = model.rmatvec(model.matvec(x0))
    with cupy.cuda.Device(gpu_id), stream as _:
        lipschitz_constant = float(cupy.abs(cupy.dot(cupy.conj(x0), x1)))

    with cupy.cuda.Device(gpu_id), stream as _:
        x0 = None
        x1 = None

    return lipschitz_constant
