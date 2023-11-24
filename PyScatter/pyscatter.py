from collections.abc import Iterable
from abc import ABC, abstractmethod
from eke import conversions
from eke import constants

from .backend import *
from . import samples

def klein_nishina(energy, scattering_angle, polarization_angle):
    """The cross section of a free electron. Energy given in eV, angles
    given in radians."""
    energy_in_joules = conversions.ev_to_J(energy)
    rescaled_energy = energy_in_joules / (constants.me*constants.c**2)

    relative_energy_change = 1. / (1. + rescaled_energy *
                                   (1. - numpy.cos(scattering_angle)))

    if polarization_angle is None:
        angle_terms = 2 * (numpy.sin(scattering_angle)**2 * 0.5)
    else:
        angle_terms = 2 * (numpy.sin(scattering_angle)**2 *
                           numpy.cos(polarization_angle)**2)
    cross_section = ((constants.re**2 * relative_energy_change**2)/2. *
                     (relative_energy_change +
                      1./relative_energy_change -
                      angle_terms))
    return cross_section


def cross_section(detector, photon_energy, polarization_angle=None):
    return klein_nishina(photon_energy,
                         detector.scattering_angle(),
                         polarization_angle)


class Source:
    def __init__(self, photon_energy, beam_energy, diameter,
                 polarization_angle=None):
        self.photon_energy = photon_energy
        self.beam_energy = beam_energy
        self.diameter = diameter
        self.polarization_angle = polarization_angle
        self.area = numpy.pi*(self.diameter/2)**2
        self.number_of_photons = (conversions.J_to_ev(self.beam_energy)
                                  / self.photon_energy)
        self.intensity = self.number_of_photons / self.area


def fourier_to_pattern(diff, detector, source):
    pattern = abs(diff)**2
    pattern *= detector.solid_angle()
    pattern *= cross_section(detector,
                             source.photon_energy,
                             source.polarization_angle)
    pattern *= source.intensity
    return pattern


def calculate_pattern(sample, detector, source, rotation=(1, 0, 0, 0)):
    if isinstance(sample, samples.MapSample):
        diff = samples.map.calculate_fourier_from_map(
            sample, detector, source.photon_energy, rotation)
    elif isinstance(sample, samples.AbstractPDB):
        diff = samples.atoms.calculate_fourier_from_pdb(
            sample, detector, source.photon_energy, rotation)
    elif isinstance(sample, samples.SphereSample):
        diff = samples.sphere.calculate_fourier_from_sphere(
            sample, detector, source.photon_energy)
    else:
        raise NotImplementedError(f"Can't calculate pattern for sample type"
                                  f"{type(sample)}")
    pattern = fourier_to_pattern(diff, detector, source)
    return pattern
    

def translate_fourier():
    pass


