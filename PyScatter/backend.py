import enum
import numpy
try:
    import cupy
except ImportError:
    cupy = None


class Backend(enum.Enum):
    """Enum coding for the available backends"""
    CPU = 1
    CUPY = 2

def set_backend(new_backend):
    """Select if calculations are run on CPU or GPU"""
    global backend, numpy, always_numpy, scipy, fft
    global ndimage, real_type, complex_type
    backend = new_backend
    if backend == Backend.CPU:
        import numpy
        always_numpy = numpy
        real_type = numpy.float32
        complex_type = numpy.complex64
    elif backend == Backend.CUPY:
        import cupy as numpy
        import numpy as always_numpy
        import functools
        real_type = functools.partial(numpy.asarray, dtype="float32")
        complex_type = functools.partial(numpy.asarray, dtype="complex64")
    else:
        raise ValueError(f"Unknown backend {backend}")


def cupy_on():
    """Check if the backend is Backend.CUPY"""
    return backend == Backend.CUPY


def cpu_on():
    """Check if the backend is Backend.CPU"""
    return backend == Backend.CPU


def cpu_array(array):
    """Convert an array to standard numpy, regardless of start type"""
    if cupy and isinstance(array, cupy.ndarray):
        return array.get()
    else:
        return array


if cupy and cupy.cuda.is_available():
    set_backend(Backend.CUPY)
    print("Using Cupy backend")
    from . import cuda_extensions
else:
    set_backend(Backend.CPU)
    print("Using CPU backend")

