from abc import ABC, abstractmethod
from typing import Iterable, Optional, Protocol, Tuple
import numpy
from numpy.typing import ArrayLike
from eke import conversions
from .backend import backend

Quaternion = Tuple[float, float, float, float]


def quaternion_to_matrix(quat: ArrayLike) -> ArrayLike:
    """Return the rotation matrix corresponding to a rotation quaternion."""
    quat = backend.real_type(numpy.array(quat))
    if len(quat.shape) < 2:
        matrix = backend.real_type(
            backend.zeros((3, 3), dtype=quat.dtype))
    else:
        matrix = backend.real_type(
            backend.zeros((len(quat), 3, 3), dtype=quat.dtype))

    matrix[..., 0, 0] = (quat[..., 0]**2 + quat[..., 1]**2
                         - quat[..., 2]**2 - quat[..., 3]**2)
    matrix[..., 0, 1] = (2 * quat[..., 1] * quat[..., 2]
                         - 2 * quat[..., 0] * quat[..., 3])
    matrix[..., 0, 2] = (2 * quat[..., 1] * quat[..., 3]
                         + 2 * quat[..., 0] * quat[..., 2])
    matrix[..., 1, 0] = (2 * quat[..., 1] * quat[..., 2]
                         + 2 * quat[..., 0] * quat[..., 3])
    matrix[..., 1, 1] = (quat[..., 0]**2 - quat[..., 1]**2
                         + quat[..., 2]**2 - quat[..., 3]**2)
    matrix[..., 1, 2] = (2 * quat[..., 2] * quat[..., 3]
                         - 2 * quat[..., 0] * quat[..., 1])
    matrix[..., 2, 0] = (2 * quat[..., 1] * quat[..., 3]
                         - 2 * quat[..., 0] * quat[..., 2])
    matrix[..., 2, 1] = (2 * quat[..., 2] * quat[..., 3]
                         + 2 * quat[..., 0] * quat[..., 1])
    matrix[..., 2, 2] = (quat[..., 0]**2 - quat[..., 1]**2
                         - quat[..., 2]**2 + quat[..., 3]**2)
    return matrix


def rotate(quat: Quaternion,
           coordinates: ArrayLike) -> ArrayLike:
    """Apply a rotation (described by a quaternion) to a set of
    coordinates.)"""
    rotation_matrix = quaternion_to_matrix(quat)

    coordinates_flat = coordinates.reshape(
        (backend.product(coordinates.shape[:-1]), 3))
    rotated_flat = coordinates_flat @ rotation_matrix.T
    return rotated_flat.reshape(coordinates.shape)


def solid_angle_rectangular_pixels(x: ArrayLike, y: ArrayLike,
                                   pixel_size: Tuple[float, float],
                                   distance: float) -> ArrayLike:
    """Calculate the solid angle of a set of rectangular pixels with
    respective centers given by x and y."""
    def omega(width: ArrayLike, height: ArrayLike
              ) -> ArrayLike:
        return 4 * numpy.arcsin(
            numpy.sin(numpy.arctan(width/2/distance)) *
            numpy.sin(numpy.arctan(height/2/distance)))

    solid_angle = (omega(2 * (x+pixel_size[0]/2),
                         2 * (y+pixel_size[1]/2)) +
                   omega(2 * (x-pixel_size[0]/2),
                         2 * (y-pixel_size[1]/2)) -
                   omega(2 * (x+pixel_size[0]/2),
                         2 * (y-pixel_size[1]/2)) -
                   omega(2 * (x-pixel_size[0]/2),
                         2 * (y+pixel_size[1]/2))) / 4
    return solid_angle


class Detector(ABC):
    """Any detector"""
    @abstractmethod
    def scattering_vector(self, photon_energy: float,
                          rotation: Quaternion = (1, 0, 0, 0)
                          ) -> ArrayLike:
        """The scattering vector for each pixel in the detector"""
        pass

    @abstractmethod
    def solid_angle(self) -> ArrayLike:
        """The solid angle occupied by each pixel in the detector"""
        pass

    @abstractmethod
    def scattering_angle(self) -> ArrayLike:
        """The scattering angle for each pixel in the detector"""
        pass


class PhysicalDetector(Detector):
    """A detector that can be interpreted as a collection of pixels with
    a location in space"""
    def scattering_vector(self, photon_energy: float,
                          rotation: Quaternion = (1, 0, 0, 0)
                          ) -> ArrayLike:
        wavelength = conversions.ev_to_m(photon_energy)

        s0 = 1/wavelength * backend.real_type(numpy.array([0, 0, 1]))

        s1_norm = numpy.sqrt(self.x**2 + self.y**2 + self.z**2)
        s1 = numpy.stack([1/wavelength * self.x / s1_norm,
                          1/wavelength * self.y / s1_norm,
                          1/wavelength * self.z / s1_norm], axis=-1)
        S = s1 - s0[numpy.newaxis, numpy.newaxis, :]
        S_rot = rotate(rotation, S)
        return S_rot

    def scattering_angle(self) -> ArrayLike:
        perp_dist = numpy.sqrt(self.x**2 + self.y**2)
        scattering_angle = numpy.arctan(perp_dist / self.z)
        return scattering_angle


class RectangularDetector(PhysicalDetector):
    """A standard rectangular detector with rectangular pixels."""
    def __init__(self, shape: Tuple[int, int],
                 pixel_size: Tuple[float, float],
                 distance: float,
                 center: Optional[Tuple[float, float]] = None):
        self.shape = shape
        self.distance = distance

        if isinstance(pixel_size, Iterable):
            self.pixel_size = pixel_size
        else:
            self.pixel_size = [pixel_size, pixel_size]

        if center is None:
            self.center = [s/2-0.5 for s in shape]
        else:
            self.center = center

        self.y, self.x = numpy.meshgrid(
            self.pixel_size[0]*(numpy.arange(self.shape[0])-self.center[0]),
            self.pixel_size[1]*(numpy.arange(self.shape[1])-self.center[1]),
            indexing="ij")

        self.y = backend.real_type(self.y)
        self.x = backend.real_type(self.x)
        self.z = self.distance*backend.real_type(numpy.ones(self.shape))

    def solid_angle(self) -> ArrayLike:
        return solid_angle_rectangular_pixels(self.x, self.y, self.pixel_size,
                                              self.distance)


class Geometry(Protocol):
    pixel_size: float

    def get_pixel_positions(self) -> ArrayLike:
        ...


class GeomDetector(PhysicalDetector):
    """A detector described by a geometry file. The solid angle calculation
    only works if the detector has rectangular pixels in a regular grid."""
    def __init__(self, geom):
        self.pixel_size = [geom.pixel_size, geom.pixel_size]

        p = geom.get_pixel_positions()
        self.x = backend.real_type(p[..., 0])
        self.y = backend.real_type(p[..., 1])
        self.z = backend.real_type(p[..., 2])
        self.distance = self.z.mean()
        self.shape = self.x.shape

    def solid_angle(self) -> ArrayLike:
        return solid_angle_rectangular_pixels(self.x, self.y, self.pixel_size,
                                              self.distance)


class FourierDetector(Detector):
    """A detector object that will retrieve that full 3D Fourier transform of
    the object"""
    def __init__(self, shape: Tuple[int, int, int], max_scatt_angle: float):
        self.shape = shape

        unscaled_fourier_max = (2*numpy.sin(max_scatt_angle/2))

        self._x, self._y, self._z = numpy.meshgrid(
            *[numpy.linspace(-unscaled_fourier_max, unscaled_fourier_max, s)
              for s in self.shape],
            indexing="ij")
        self._x = backend.real_type(self._x)
        self._y = backend.real_type(self._y)
        self._z = backend.real_type(self._z)

    def scattering_angle(self) -> ArrayLike:
        """The scattering angle that each pixel would have if it was part of
        a physical detector"""
        dist = numpy.sqrt(self._x**2 + self._y**2 + self._z**2)
        return 2 * numpy.arcsin(dist/2)

    def scattering_vector(self, photon_energy: float,
                          rotation: Quaternion = (1, 0, 0, 0)
                          ) -> ArrayLike:
        wavelength = conversions.ev_to_m(photon_energy)
        # S = 1/wavelength * numpy.array((self._x, self._y, self._z))
        S = 1/wavelength * numpy.stack((self._x, self._y, self._z), axis=3)
        S_rot = rotate(rotation, S)
        return S_rot

    def solid_angle(self) -> ArrayLike:
        """The solid angle is assumed to be unity for all pixels, since
        it doesn't make much sense for a 3D Fourier space"""
        return numpy.ones(self.shape)
