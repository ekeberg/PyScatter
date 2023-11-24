import numpy
from eke import elements
from eke import constants
from ..pyscatter import *

class SphereSample:
    def __init__(self, diameter, material):
        self.diameter = diameter
        self.material = material


def calculate_fourier_from_sphere(sample, detector, photon_energy):
    S = numpy.linalg.norm(detector.scattering_vector(photon_energy, (1, 0, 0, 0)), axis=-1)

    f_sum = 0
    for element, ratio in sample.material.element_mass_ratios().items():
        ratio /= sample.material.element_mass_ratio_sum()
        f = elements.get_scattering_factor(element, photon_energy)
        natoms_per_m3 = (sample.material.material_density()
                            * ratio
                            / (elements.ATOMIC_MASS[element] * constants.u))
        f_sum += f * natoms_per_m3

    # scaling = numpy.pi**2 * sample.diameter**3 * f_sum
    # scaling = f_sum * (4/3 * numpy.pi * sample.diameter**3) * (numpy.pi)
    radius = sample.diameter/2
    scaling = f_sum * 4 * numpy.pi * radius**3

    s = 2*numpy.pi*radius*S
    structure = (numpy.sin(s) - s*numpy.cos(s))/s**3
    return scaling*structure
