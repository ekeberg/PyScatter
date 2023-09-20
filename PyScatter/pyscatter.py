# import numpy
import enum

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

import Bio.PDB
import pickle
from eke import conversions
from eke import constants
from eke import rotmodule
from eke import elements

try:
    import nfft
except ImportError:
    nfft = None

class AbstractPDB:
    def __init__(self, filename):
        pass

    def unique_elements(self):
        # return numpy.unique(self.elements)
        return list(set(self.elements))

class SimplePDB(AbstractPDB):
    """Uses the Biopython parser"""
    def __init__(self, filename):
        self._parser = Bio.PDB.PDBParser(QUIET=False)
        self.struct = self._parser.get_structure("foo", filename)

        atoms = [a for a in self.struct.get_atoms()]
        self.natoms = len(atoms)
        # self.coords = real_type(numpy.zeros((self.natoms, 3)))
        self.coords = always_numpy.zeros((self.natoms, 3))
        self.elements = []
        # self.occupancy = real_type(numpy.zeros(self.natoms))
        self.occupancy = always_numpy.zeros(self.natoms)

        for i, a in enumerate(atoms):
            self.coords[i, :] = a.get_coord()*1e-10 # Convert to m
            self.elements.append(a.element)
            self.occupancy[i] = a.occupancy

        self.coords = real_type(self.coords)
        self.occupancy = real_type(self.occupancy)


class SloppyPDB(AbstractPDB):
    """Reads all atoms, regardless of if it makes sense or not"""
    def __init__(self, filename):

        self.coords = []
        self.elements = []
        self.occupancy = []
        
        with open(filename) as f:
            for l in f.readlines():
                atom = self._parse_line(l)

                if atom is not None:
                    self.coords.append(atom["coord"])
                    self.elements.append(atom["element"])
                    self.occupancy.append(atom["occupancy"])

        self.coords = real_type(numpy.array(self.coords))
        self.occupancy = real_type(numpy.array(self.occupancy))
        self.natoms = len(self.elements)
                    
    def _parse_line(self, l):
        if (l[:4].upper() == "ATOM" or
            l[:6].upper() == "HETATM"):
            atom = {}
            atom["element"] = l[77:78+1].strip()
            atom["coord"] = (float(l[31:38+1].strip())*1e-10,
                             float(l[39:46+1].strip())*1e-10,
                             float(l[47:54+1].strip())*1e-10)
            atom["occupancy"] = float(l[55:60+1].strip())
            return atom
        else:
            return None


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
            raise ValueError("Can not get shape of sample with no density added to it")
        return self.maps[0].shape

def quaternion_to_matrix(quat):
    """Dummy docstring"""
    quat = real_type(numpy.array(quat))
    if len(quat.shape) < 2:
        matrix = real_type(numpy.zeros((3, 3), dtype=quat.dtype))
    else:
        matrix = real_type(numpy.zeros((len(quat), 3, 3), dtype=quat.dtype))

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
    # return rotation_matrix.dot(coordinates.T).T
    # return rotation_matrix.dot(coordinates)
    coordinates_flat = coordinates.reshape((coordinates.shape[0], always_numpy.product(coordinates.shape[1:])))
    rotated_flat = rotation_matrix.dot(coordinates_flat)
    return rotated_flat.reshape(coordinates.shape)


class RectangularDetector:
    def __init__(self, shape, pixel_size, distance, center=None):
        self.shape = shape
        self.distance = distance
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
        
    def scattering_vector(self, photon_energy, rotation=(1, 0, 0, 0)):
        wavelength = conversions.ev_to_m(photon_energy)

        s0 = 1/wavelength * real_type(numpy.array([0, 0, 1]))

        s1_norm = numpy.sqrt(self.x**2 + self.y**2 + self.z**2)
        s1 = numpy.array([1/wavelength * self.x / s1_norm,
                          1/wavelength * self.y / s1_norm,
                          1/wavelength * self.z / s1_norm])
        S = s1 - s0[:, numpy.newaxis, numpy.newaxis]
        S_rot = rotate(rotation, S)
        return S_rot

    def scattering_angle(self):
        in_plane_distance = numpy.sqrt(self.x**2 + self.y**2)
        scattering_angle = numpy.arctan(in_plane_distance / self.distance)
        return scattering_angle

    def old_solid_angle(self):
        area = self.pixel_size[0] * self.pixel_size[1]
        distance = numpy.sqrt(self.x**2 + self.y**2 + self.z**2)
        return area / distance

    def _omega(self, width, height):
        return 4 * numpy.arcsin(numpy.sin(numpy.arctan(width/2/self.distance)) *
                                numpy.sin(numpy.arctan(height/2/self.distance)))
    
    def solid_angle(self):
        sa = (self._omega(2 * (self.x+self.pixel_size[0]/2),
                          2 * (self.y+self.pixel_size[1]/2)) +
              self._omega(2 * (self.x-self.pixel_size[0]/2),
                          2 * (self.y-self.pixel_size[1]/2)) -
              self._omega(2 * (self.x+self.pixel_size[0]/2),
                          2 * (self.y-self.pixel_size[1]/2)) -
              self._omega(2 * (self.x-self.pixel_size[0]/2),
                          2 * (self.y+self.pixel_size[1]/2))) / 4
        return sa


class FourierDetector:
    def __init__(self, shape, max_scatt_angle):
        self.shape = shape

        unscaled_fourier_max = (2*numpy.sin(max_scatt_angle/2))

        self._x, self._y, self._z = numpy.meshgrid(
            numpy.linspace(-unscaled_fourier_max, unscaled_fourier_max, shape[0]),
            numpy.linspace(-unscaled_fourier_max, unscaled_fourier_max, shape[1]),
            numpy.linspace(-unscaled_fourier_max, unscaled_fourier_max, shape[2]),
            indexing="ij")
        self._x = real_type(self._x)
        self._y = real_type(self._y)
        self._z = real_type(self._z)

    def scattering_angle(self):
        # return 2* numpy.arcsin(1/numpy.sqrt(self._x**2 + self._y**2 + self._z**2)/2)
        return 2 * numpy.arcsin((numpy.sqrt(self._x**2 + self._y**2 + self._z**2)/2))

    def scattering_vector(self, photon_energy, rotation=(1, 0, 0, 0)):
        wavelength = conversions.ev_to_m(photon_energy)
        S = 1/wavelength * numpy.array((self._x, self._y, self._z))
        S_rot = rotate(rotation, S)
        return S_rot

    def solid_angle(self):
        return numpy.ones(self.shape)


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

import pathlib
STRUCTURE_FACTOR_TABLE = pickle.load(open(pathlib.Path(__file__).parent / "structure_factors.p", "rb"))
                                          

class StructureFactors:
    def __init__(self):        
        # self._table = pickle.load(open("structure_factors.p", "rb"))
        self._table = STRUCTURE_FACTOR_TABLE
        self.precalculated = {}

    def structure_factor(self, element, s):
        if element not in self._table:
            raise ValueError("Element {} is not recognized".format(element))
        s_A_half = s*1e-10/2
        p = self._table[element]
        return (p[0]*numpy.exp(-p[1]*s_A_half**2) +
                p[2]*numpy.exp(-p[3]*s_A_half**2) +
                p[4]*numpy.exp(-p[5]*s_A_half**2) +
                p[6]*numpy.exp(-p[7]*s_A_half**2) + p[8])

    def precalculate_for_element(self, element, S_array):
        self.precalculated[element] = real_type(self.structure_factor(element, S_array))


class Source:
    def __init__(self, photon_energy, beam_energy, diameter, polarization_angle=None):
        self.photon_energy = photon_energy
        self.beam_energy = beam_energy
        self.diameter = diameter
        self.polarization_angle = polarization_angle
        self.area = numpy.pi*(self.diameter/2)**2
        self.number_of_photons = (conversions.J_to_ev(self.beam_energy)
                                  / self.photon_energy)
        self.intensity = self.number_of_photons / self.area


def calculate_fourier_from_pdb(pdb, detector, photon_energy, rotation=(1, 0, 0, 0)):
    if cupy_on():
        return calculate_fourier_from_pdb_cuda(pdb, detector, photon_energy, rotation)
    else:
        return calculate_fourier_from_pdb_cpu(pdb, detector, photon_energy, rotation)

def calculate_fourier_from_pdb_cpu(pdb, detector, photon_energy, rotation=(1, 0, 0, 0)):
    S = detector.scattering_vector(photon_energy, rotation)
    
    sf = StructureFactors()
    for element in pdb.unique_elements():
        sf.precalculate_for_element(element, numpy.linalg.norm(S, axis=0))
    
    diff = complex_type(numpy.zeros(detector.shape))

    for coord, occupancy, element in zip(pdb.coords, pdb.occupancy, pdb.elements):
        coord_slice = (slice(None), ) + (None, )*len(S.shape[1:])
        dotp = (coord[coord_slice] * S).sum(axis=0)
        diff += (sf.precalculated[element] * occupancy *
                 numpy.exp(2j * numpy.pi * dotp))
    return diff


def calculate_fourier_from_pdb_cuda(pdb, detector, photon_energy, rotation=(1, 0, 0, 0)):
    S = detector.scattering_vector(photon_energy, rotation)

    sf = StructureFactors()
    for element in pdb.unique_elements():
        sf.precalculate_for_element(element, numpy.linalg.norm(S, axis=0))

    diff = complex_type(numpy.zeros(detector.shape))

    unique_elements = list(always_numpy.unique(pdb.elements))
    for element in unique_elements:
        selection = numpy.asarray(always_numpy.array(pdb.elements) == element, dtype="bool")
        element_coords = cupy.ascontiguousarray(pdb.coords[selection], dtype="float32")
        element_occupancy = cupy.ascontiguousarray(pdb.occupancy[selection], dtype="float32")

        element_diff = numpy.zeros_like(diff)
        cuda_extensions.calculate_scattering(element_diff, S, element_coords, element_occupancy)
        diff += sf.precalculated[element] * element_diff
    return diff

def calculate_pattern_from_pdb(pdb, detector, source, rotation=(1, 0, 0, 0)):
    diff = calculate_fourier_from_pdb(
        pdb, detector, source.photon_energy, rotation)
    pattern = abs(diff)**2
    pattern *= detector.solid_angle()
    pattern *= cross_section(detector, source.photon_energy, source.polarization_angle)
    pattern *= source.intensity
    return pattern


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

def calculate_fourier_from_map(sample, detector, photon_energy, rotation=(1, 0, 0, 0)):
    if nfft is None:
        raise RuntimeError("Could not import nfft")

    total_density_map = numpy.zeros(sample.shape(), dtype="complex128")

    for material, material_map in zip(sample.materials, sample.maps):
        total_density_map += get_scat_map(material_map,
                                          material,
                                          sample.pixel_size,
                                          photon_energy)

    S = detector.scattering_vector(photon_energy, rotation)
    diff = nfft.nfft(total_density_map, sample.pixel_size, S.reshape((3, numpy.product(detector.shape))).T).reshape(detector.shape)
    return diff

def calculate_pattern(sample, detector, source, rotation=(1, 0, 0, 0)):
    if isinstance(sample, MapSample):
        diff = calculate_fourier_from_map(
            sample, detector, source.photon_energy, rotation)
    elif isinstance(sample, AbstractPDB):
        diff = calculate_fourier_from_pdb(
            sample, detector, source.photon_energy, rotation)
    pattern = abs(diff)**2
    pattern *= detector.solid_angle()
    pattern *= cross_section(detector, source.photon_energy, source.polarization_angle)
    pattern *= source.intensity
    return pattern
    

def translate_fourier():
    pass


