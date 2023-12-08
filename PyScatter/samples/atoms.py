from dataclasses import dataclass
import pickle
from typing import Optional, Protocol, Tuple
from numpy.typing import ArrayLike
import Bio.PDB
import pathlib
import importlib.resources as pkg_resources
from ..backend import cuda_tools
from ..backend import backend

Quaternion = Tuple[float, float, float, float]


class Detector(Protocol):
    """A detector that can calculate the scattering vector for a given
    photon energy and rotation"""
    def scattering_vector(self, photon_energy: float,
                          rotation: Quaternion = (1, 0, 0, 0)) -> ArrayLike:
        """Calculate the scattering vector for a given photon energy and
        rotation"""
        ...


STRUCTURE_FACTOR_TABLE = pickle.loads(
    pkg_resources.read_binary("PyScatter", "structure_factors.p"))


class StructureFactors:
    """Class for calculating structure factors using the 9 parameter formula"""
    def __init__(self):
        # self._table = pickle.load(open("structure_factors.p", "rb"))
        self._table = STRUCTURE_FACTOR_TABLE
        self.precalculated = {}

    def structure_factor(self, element: str, s: ArrayLike
                         ) -> ArrayLike:
        if element not in self._table:
            raise ValueError("Element {} is not recognized".format(element))
        s_A_half = s*1e-10/2
        p = self._table[element]
        return (p[0]*backend.exp(-p[1]*s_A_half**2) +
                p[2]*backend.exp(-p[3]*s_A_half**2) +
                p[4]*backend.exp(-p[5]*s_A_half**2) +
                p[6]*backend.exp(-p[7]*s_A_half**2) + p[8])

    def precalculate_for_element(self, element: str, S_array: ArrayLike
                                 ) -> None:
        self.precalculated[element] = backend.real_type(
            self.structure_factor(element, S_array))


@dataclass
class AtomsSample:
    """A sample described by a list of atoms"""
    natoms: int
    coords: backend.ndarray
    elements: list
    occupancy: backend.ndarray
    bfactor: backend.ndarray = None

    def unique_elements(self) -> list:
        """Returns a list of the elements present in the sample"""
        return list(set(self.elements))


def read_pdb(filename: pathlib.Path, use_bfactor: bool = False,
             quiet: bool = True) -> AtomsSample:
    """Uses the Biopython parser to read a .pdb or .cif file"""
    if pathlib.Path(filename).suffix.lower() == ".cif":
        parser = Bio.PDB.MMCIFParser(QUIET=quiet)
    else:
        parser = Bio.PDB.PDBParser(QUIET=quiet)

    struct = parser.get_structure("foo", filename)

    atoms = [a for a in struct.get_atoms()]
    natoms = len(atoms)
    coords = backend.real_type(backend.zeros((natoms, 3)))
    occupancy = backend.real_type(backend.zeros(natoms))
    elements = []

    if use_bfactor:
        bfactor = backend.real_type(backend.zeros(natoms))
    else:
        bfactor = None

    for i, a in enumerate(atoms):
        coords[i, :] = backend.real_type(a.get_coord()*1e-10)  # Convert to m
        elements.append(a.element.capitalize())
        occupancy[i] = a.occupancy
        if use_bfactor:
            bfactor[i] = a.bfactor * 1e-20  # Convert to m^2

    return AtomsSample(natoms, coords, elements, occupancy, bfactor)


def read_pdb_sloppy(filename: pathlib.Path, use_bfactor: bool = False
                    ) -> AtomsSample:
    """Reads all atoms of a .pdb file, regardless of if it makes sense or
    not"""
    use_bfactor = bool(use_bfactor)

    coords = []
    elements = []
    occupancy = []
    if use_bfactor:
        bfactor = []
    else:
        bfactor = None

    def parse_line(line):
        if (line[:4].upper() == "ATOM" or
                line[:6].upper() == "HETATM"):
            atom = {}
            atom["element"] = line[77:78+1].strip()
            atom["coord"] = (float(line[31:38+1].strip())*1e-10,
                             float(line[39:46+1].strip())*1e-10,
                             float(line[47:54+1].strip())*1e-10)
            atom["occupancy"] = float(line[55:60+1].strip())
            if use_bfactor:
                atom["bfactor"] = float(line[61:66+1].strip()) * 1e-20
            return atom
        else:
            return None

    with open(filename) as f:
        for line in f.readlines():
            atom = parse_line(line)

            if atom is not None:
                coords.append(atom["coord"])
                elements.append(atom["element"])
                occupancy.append(atom["occupancy"])
                if use_bfactor:
                    bfactor.append(atom["bfactor"])

    coords = backend.real_type(backend.array(coords))
    occupancy = backend.real_type(backend.array(occupancy))
    if use_bfactor:
        bfactor = backend.real_type(backend.array(bfactor))
    natoms = len(elements)
    return AtomsSample(natoms, coords, elements, occupancy, bfactor)


def calculate_fourier(sample: AtomsSample, detector: Detector,
                      photon_energy: float,
                      rotation: Quaternion = (1, 0, 0, 0)
                      ) -> ArrayLike:
    if backend.is_cupy():
        return calculate_fourier_cuda(
            sample, detector, photon_energy, rotation)
    else:
        return calculate_fourier_cpu(
            sample, detector, photon_energy, rotation)


def calculate_fourier_cpu(sample: AtomsSample, detector: Detector,
                          photon_energy: float,
                          rotation: Quaternion = (1, 0, 0, 0)
                          ) -> ArrayLike:
    S = detector.scattering_vector(photon_energy, rotation)

    sf = StructureFactors()
    for element in sample.unique_elements():
        sf.precalculate_for_element(element, backend.linalg.norm(S, axis=-1))

    diff = backend.complex_type(backend.zeros(detector.shape))

    use_bfactor = sample.bfactor is not None

    if use_bfactor:
        S_abs = backend.linalg.norm(S, axis=-1)

    atom_iterator = zip(sample.coords, sample.occupancy, sample.elements)
    for coord, occupancy, element in atom_iterator:
        # coord_slice = (slice(None), ) + (None, )*len(S.shape[:-1])
        # dotp = (coord[coord_slice] * S).sum(axis=0)
        # dotp = (coord * S).sum(axis=-1)
        dotp = S @ coord

        atom_diff = (sf.precalculated[element] * occupancy *
                     backend.exp(2j * backend.pi * dotp))

        if use_bfactor:
            atom_diff *= backend.exp(-sample.bfactor * S_abs**2 * 0.25)

        diff += atom_diff
    return diff


if backend.is_cupy():
    __kernels = cuda_tools.import_cuda_file(
        'atoms_cuda.cu', ['calculate_scattering'])

    def _calculate_scattering(element_diff: ArrayLike,
                              S: ArrayLike,
                              element_coords: ArrayLike,
                              element_occupancy: ArrayLike,
                              bfactor: Optional[ArrayLike] = None) -> None:
        if bfactor is None:
            use_bfactor = False
        else:
            use_bfactor = True

        nthreads = 256
        nblocks = (element_diff.size - 1) // nthreads + 1

        arguments = (element_diff, element_diff.size, S,
                     element_coords, element_coords.shape[0],
                     element_occupancy, use_bfactor, bfactor)
        __kernels["calculate_scattering"]((nblocks, ), (nthreads, ), arguments)

    def calculate_fourier_cuda(sample: AtomsSample,
                               detector: Detector,
                               photon_energy: float,
                               rotation: Quaternion = (1, 0, 0, 0)
                               ) -> ArrayLike:
        S = detector.scattering_vector(photon_energy, rotation)

        sf = StructureFactors()
        for element in sample.unique_elements():
            sf.precalculate_for_element(
                element, backend.linalg.norm(S, axis=-1))

        diff = backend.complex_type(backend.zeros(detector.shape))

        unique_elements = sample.unique_elements()
        for element in unique_elements:
            element_array = backend.asnumpy(sample.elements)
            selection = backend.asarray(element_array == element,
                                        dtype="bool")
            element_coords = backend.ascontiguousarray(
                sample.coords[selection], dtype="float32")
            element_occupancy = backend.ascontiguousarray(
                sample.occupancy[selection], dtype="float32")

            use_bfactor = sample.bfactor is not None

            if use_bfactor:
                element_bfactor = backend.ascontiguousarray(
                    sample.bfactor[selection], dtype="float32")
            else:
                element_bfactor = None

            element_diff = backend.zeros_like(diff)
            _calculate_scattering(
                element_diff, S, element_coords, element_occupancy,
                bfactor=element_bfactor)
            diff += sf.precalculated[element] * element_diff
        return diff
