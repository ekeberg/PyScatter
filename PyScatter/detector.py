from abc import ABC, abstractmethod
from typing import Iterable

import numpy
from eke import conversions

from .pyscatter import *


def quaternion_to_matrix(quat):
    """Dummy docstring"""
    quat = real_type(numpy.array(quat))
    if len(quat.shape) < 2:
        matrix = real_type(
            numpy.zeros((3, 3), dtype=quat.dtype))
    else:
        matrix = real_type(
            numpy.zeros((len(quat), 3, 3), dtype=quat.dtype))

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


def rotate(quat, coordinates):
    rotation_matrix = quaternion_to_matrix(quat)

    coordinates_flat = coordinates.reshape(
        (always_numpy.product(coordinates.shape[:-1]), 3))
    rotated_flat = coordinates_flat @ rotation_matrix.T
    return rotated_flat.reshape(coordinates.shape)


class Detector(ABC):

    @abstractmethod
    def scattering_vector(self, photon_energy, rotation=(1, 0, 0, 0)):
        pass

    @abstractmethod
    def solid_angle(self):
        pass

class PhysicalDetector(Detector):
    def scattering_vector(self, photon_energy, rotation=(1, 0, 0, 0)):
        wavelength = conversions.ev_to_m(photon_energy)

        s0 = 1/wavelength * real_type(numpy.array([0, 0, 1]))

        s1_norm = numpy.sqrt(self.x**2 + self.y**2 + self.z**2)
        s1 = numpy.stack([1/wavelength * self.x / s1_norm,
                          1/wavelength * self.y / s1_norm,
                          1/wavelength * self.z / s1_norm], axis=-1)
        S = s1 - s0[numpy.newaxis, numpy.newaxis, :]
        S_rot = rotate(rotation, S)
        return S_rot
    
    def scattering_angle(self):
        perp_dist = numpy.sqrt(self.x**2 + self.y**2)
        scattering_angle = numpy.arctan(perp_dist / self.z)
        return scattering_angle

    @abstractmethod
    def solid_angle():
        pass

class RectangularDetector(PhysicalDetector):
    def __init__(self, shape, pixel_size, distance, center=None):
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
        self.y = real_type(self.y)
        self.x = real_type(self.x)
        self.z = self.distance*real_type(numpy.ones(self.shape))
        
    # def scattering_angle(self):
    #     in_plane_distance = numpy.sqrt(self.x**2 + self.y**2)
    #     scattering_angle = numpy.arctan(in_plane_distance / self.distance)
    #     return scattering_angle

    def solid_angle(self):
        return solid_angle_rectangular_pixels(self.x, self.y, self.pixel_size,
                                              self.distance)


def solid_angle_rectangular_pixels(x, y, pixel_size, distance):
    def omega(width, height):
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


class GeomDetector(PhysicalDetector):
    def __init__(self, geom):
        self.pixel_size = [geom.pixel_size, geom.pixel_size]        
        
        p = geom.get_pixel_positions()
        self.x = real_type(p[..., 0])
        self.y = real_type(p[..., 1])
        self.z = real_type(p[..., 2])
        self.distance = self.z.mean()
        self.shape = self.x.shape

    def solid_angle(self):
        return solid_angle_rectangular_pixels(self.x, self.y, self.pixel_size,
                                              self.distance)


class FourierDetector(Detector):
    def __init__(self, shape, max_scatt_angle):
        self.shape = shape

        unscaled_fourier_max = (2*numpy.sin(max_scatt_angle/2))

        self._x, self._y, self._z = numpy.meshgrid(
            *[numpy.linspace(-unscaled_fourier_max, unscaled_fourier_max, s)
              for s in self.shape],
            indexing="ij")
        self._x = real_type(self._x)
        self._y = real_type(self._y)
        self._z = real_type(self._z)

    def scattering_angle(self):
        dist = numpy.sqrt(self._x**2 + self._y**2 + self._z**2)
        return 2 * numpy.arcsin(dist/2)

    def scattering_vector(self, photon_energy, rotation=(1, 0, 0, 0)):
        wavelength = conversions.ev_to_m(photon_energy)
        # S = 1/wavelength * numpy.array((self._x, self._y, self._z))
        S = 1/wavelength * numpy.stack((self._x, self._y, self._z), axis=3)
        S_rot = rotate(rotation, S)
        return S_rot

    def solid_angle(self):
        return numpy.ones(self.shape)
