import numpy
from typing import Tuple, Optional
from numpy.typing import ArrayLike
from ..backend import backend
from ..pyscatter import detector
from eke import elements
from eke import constants

try:
    import nfft
except ImportError:
    nfft = None

Quaternion = Tuple[float, float, float, float]


class MapSample:
    """A sample described by a pixel map of material densities"""
    def __init__(self, pixel_size: Tuple[int, int, int],
                 distribution: Optional[ArrayLike] = None,
                 material: Optional[elements.Material] = None):
        self.pixel_size = pixel_size
        self.maps = []
        self.materials = []
        if distribution is not None and material is not None:
            self.add_map(distribution, material)

    def add_map(self, distribution, material):
        """Add a map of a material. Use this to build up your sample"""
        self.maps.append(distribution)
        self.materials.append(material)

    @property
    def shape(self) -> Tuple[int, int, int]:
        if len(self.maps) == 0:
            raise ValueError("Can not get shape of sample since there is no "
                             "density added to it")
        return self.maps[0].shape


def get_scat_map(distribution: ArrayLike, material: elements.Material,
                 pixel_size: float, photon_energy: float
                 ) -> ArrayLike:
    """Calculate the scattering map for a given material and distribution.
    If the distribution is 1, it is using the density of the material, and
    any variation will imply a scaling of the density."""
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
    scat_map = backend.real_type(distribution) * f_sum
    return scat_map


def calculate_fourier(sample: MapSample, detector: detector.Detector,
                      photon_energy: float,
                      rotation: Quaternion = (1, 0, 0, 0)
                      ) -> ArrayLike:
    if nfft is None:
        raise RuntimeError("Could not import nfft")

    total_density_map = backend.complex_type(backend.zeros(sample.shape))

    for material, material_map in zip(sample.materials, sample.maps):
        total_density_map += get_scat_map(material_map,
                                          material,
                                          sample.pixel_size,
                                          photon_energy)

    S = detector.scattering_vector(photon_energy, rotation)
    S_flat = S.reshape((numpy.product(detector.shape), 3))
    diff = nfft.nfft(backend.asnumpy(total_density_map), sample.pixel_size, backend.asnumpy(S_flat))
    diff = diff.reshape(detector.shape)
    diff = backend.complex_type(diff)
    return diff
