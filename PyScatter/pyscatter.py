from typing import Any, Optional, Tuple
from eke import conversions
from eke import constants
from numpy.typing import ArrayLike

from .backend import backend
from . import detector
from . import samples


Quaternion = Tuple[float, float, float, float]


def cross_section(energy: float, scattering_angle: ArrayLike,
                  polarization_angle: Optional[float] = None
                  ) -> ArrayLike:
    """The cross section of a free electron. Energy given in eV, angles
    given in radians. Calculated using the Klein-Nishina formula."""
    energy_in_joules = conversions.ev_to_J(energy)
    rescaled_energy = energy_in_joules / (constants.me*constants.c**2)

    relative_energy_change = 1. / (1. + rescaled_energy *
                                   (1. - backend.cos(scattering_angle)))

    if polarization_angle is None:
        angle_terms = 2 * (backend.sin(scattering_angle)**2 * 0.5)
    else:
        angle_terms = 2 * (backend.sin(scattering_angle)**2 *
                           backend.cos(polarization_angle)**2)
    cross_section = ((constants.re**2 * relative_energy_change**2)/2. *
                     (relative_energy_change +
                      1./relative_energy_change -
                      angle_terms))
    return cross_section


class Source:
    def __init__(self, photon_energy: float, beam_energy: float,
                 diameter: float, polarization_angle: Optional[float] = None):
        self.photon_energy = photon_energy
        self.beam_energy = beam_energy
        self.diameter = diameter
        self.polarization_angle = polarization_angle
        self.area = backend.pi*(self.diameter/2)**2
        self.number_of_photons = (conversions.J_to_ev(self.beam_energy)
                                  / self.photon_energy)
        self.intensity = self.number_of_photons / self.area


def fourier_to_pattern(diff: ArrayLike, detector: detector.Detector,
                       source: Source) -> ArrayLike:
    """Converts the diffraction pattern to scattered intensities. This also
    scales with respect to detector solid angle, the cross section at different
    scattering angles and the intensity of the source."""
    pattern = abs(diff)**2
    pattern *= detector.solid_angle()
    pattern *= cross_section(source.photon_energy,
                             detector.scattering_angle(),
                             source.polarization_angle)
    pattern *= source.intensity
    return pattern


def calculate_pattern(sample: Any, detector: detector.Detector, source: Source,
                      rotation: Quaternion = (1, 0, 0, 0)
                      ) -> ArrayLike:
    """Calculates the diffraction pattern"""
    if isinstance(sample, samples.MapSample):
        diff = samples.map.calculate_fourier(
            sample, detector, source.photon_energy, rotation)
    elif isinstance(sample, samples.AtomsSample):
        diff = samples.atoms.calculate_fourier(
            sample, detector, source.photon_energy, rotation)
    elif isinstance(sample, samples.SphereSample):
        diff = samples.sphere.calculate_fourier(
            sample, detector, source.photon_energy)
    else:
        raise NotImplementedError(f"Can't calculate pattern for sample type"
                                  f"{type(sample)}")
    pattern = fourier_to_pattern(diff, detector, source)
    return pattern


def translate_fourier(fourier_map: ArrayLike,
                      scattering_vector: ArrayLike,
                      translation: Tuple[float, float, float]
                      ) -> ArrayLike:
    """Translate a particle by multiplying it's Fourier transform (fourier_map)
    by a phase ramp."""
    dotp = scattering_vector @ translation
    ramp = backend.exp(2.j * backend.pi * dotp)
    return fourier_map * ramp
