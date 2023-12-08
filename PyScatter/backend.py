import enum
import numpy
try:
    import cupy
except ImportError:
    cupy = None


class Backend(enum.Enum):
    NUMPY = numpy
    CUPY = cupy


class BackendContext:
    def __init__(self):
        self._backend = Backend.NUMPY

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend):
        if backend in Backend:
            self._backend = backend
        else:
            raise ValueError(f"Unknown backend {backend}")

    def __enter__(self):
        self._old_backend = self._backend
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._backend = self._old_backend

    def __getattr__(self, name):
        return getattr(self._backend.value, name)

    def is_numpy(self):
        return self._backend == Backend.NUMPY

    def is_cupy(self):
        return self._backend == Backend.CUPY

    def asnumpy(self, array):
        if self._backend == Backend.CUPY and isinstance(array, cupy.ndarray):
            return array.get()
        else:
            return array

    def real_type(self, array):
        if self._backend == Backend.CUPY:
            return cupy.asarray(array, dtype="float32")
        else:
            return numpy.asarray(array, dtype="float32")

    def complex_type(self, array):
        if self._backend == Backend.CUPY:
            return cupy.asarray(array, dtype="complex64")
        else:
            return numpy.asarray(array, dtype="complex64")


backend = BackendContext()

if cupy and cupy.cuda.is_available():
    backend.backend = Backend.CUPY
    from . import cuda_tools
    print("Using Cupy backend")
else:
    backend.backend = Backend.NUMPY
    cuda_tools = None
    print("Using Numpy backend")

