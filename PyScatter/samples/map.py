import numpy
from ..pyscatter import *
from eke import elements
from eke import constants

try:
    import nfft
except ImportError:
    nfft = None



class MapSample:
    def __init__(self, pixel_size, distribution=None, material=None):
        self.pixel_size = pixel_size
        self.maps = []
        self.materials = []
        if distribution is not None and material is not None:
            self.add_map(distribution, material)

    def add_map(self, distribution, material):
        self.maps.append(distribution)
        self.materials.append(material)

    def shape(self):
        if len(self.maps) == 0:
            raise ValueError("Can not get shape of sample since there is no "
                             "density added to it")
        return self.maps[0].shape


def get_scat_map(distribution, material, pixel_size, photon_energy):
    f_sum = 0
    for element, ratio in material.element_mass_ratios().items():
        ratio /= material.element_mass_ratio_sum()
        f = elements.get_scattering_factor(element, photon_energy)
        # natoms = density * volume / element_weight * elemet_ratio
        natoms_per_pixel = (material.material_density()
                            * pixel_size**3
                            * ratio
                            / (elements.ATOMIC_MASS[element] * constants.u))
        f_sum += f * natoms_per_pixel
    scat_map = distribution * f_sum
    return scat_map


def calculate_fourier_from_map(sample, detector, photon_energy,
                               rotation=(1, 0, 0, 0)):
    if nfft is None:
        raise RuntimeError("Could not import nfft")

    total_density_map = numpy.zeros(sample.shape(), dtype="complex128")

    for material, material_map in zip(sample.materials, sample.maps):
        total_density_map += get_scat_map(material_map,
                                          material,
                                          sample.pixel_size,
                                          photon_energy)

    S = detector.scattering_vector(photon_energy, rotation)
    # S_transpose = S.reshape((3, numpy.product(detector.shape))).T
    S_flat = S.reshape((numpy.product(detector.shape), 3))
    diff = nfft.nfft(total_density_map, sample.pixel_size, S_flat)
    diff = diff.reshape(detector.shape)
    return diff
