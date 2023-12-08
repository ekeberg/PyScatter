from dataclasses import dataclass
from typing import Protocol, Tuple
from ..backend import backend
from numpy.typing import ArrayLike
from eke import elements
from eke import constants

Quaternion = Tuple[float, float, float, float]


class Detector(Protocol):
    """A detector that can calculate the scattering vector for a given
    photon energy and rotation"""
    def scattering_vector(self, photon_energy: float,
                          rotation: Quaternion = (1, 0, 0, 0)) -> ArrayLike:
        """Calculate the scattering vector for a given photon energy and
        rotation"""
        ...


@dataclass
class SphereSample:
    """A spherical sample"""
    diameter: float
    material: elements.Material


def calculate_fourier(sample: SphereSample, detector: Detector,
                      photon_energy: float) -> ArrayLike:
    S = detector.scattering_vector(photon_energy, (1, 0, 0, 0))
    S_norm = backend.linalg.norm(S, axis=-1)

    f_sum = 0
    for element, ratio in sample.material.element_mass_ratios().items():
        ratio /= sample.material.element_mass_ratio_sum()
        f = elements.get_scattering_factor(element, photon_energy)
        natoms_per_m3 = (sample.material.material_density()
                         * ratio
                         / (elements.ATOMIC_MASS[element] * constants.u))
        f_sum += f * natoms_per_m3

    radius = sample.diameter/2
    scaling = f_sum * 4 * backend.pi * radius**3

    s = 2*backend.pi*radius*S_norm
    structure = (backend.sin(s) - s*backend.cos(s))/s**3
    return scaling*structure
