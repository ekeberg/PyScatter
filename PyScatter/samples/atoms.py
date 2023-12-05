from dataclasses import dataclass
import pickle
from typing import Optional, Tuple
import Bio.PDB
import pathlib
# from ..pyscatter import *
from .. import pyscatter
from ..pyscatter import detector
numpy = pyscatter.numpy
from numpy.typing import ArrayLike

import importlib.resources as pkg_resources
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
        return (p[0]*numpy.exp(-p[1]*s_A_half**2) +
                p[2]*numpy.exp(-p[3]*s_A_half**2) +
                p[4]*numpy.exp(-p[5]*s_A_half**2) +
                p[6]*numpy.exp(-p[7]*s_A_half**2) + p[8])

    def precalculate_for_element(self, element: str, S_array: ArrayLike
                                 ) -> None:
        self.precalculated[element] = pyscatter.real_type(
            self.structure_factor(element, S_array))
        

@dataclass
class AtomsSample:
    """A sample described by a list of atoms"""
    natoms: int
    coords: numpy.ndarray
    elements: list
    occupancy: numpy.ndarray
    bfactor: numpy.ndarray = None

    def unique_elements(self) -> list:
        """Returns a list of the elements present in the sample"""
        return list(set(self.elements))


def read_pdb(filename: pathlib.Path, use_bfactor: bool=False, quiet: bool=True
             ) -> AtomsSample:
    """Uses the Biopython parser to read a .pdb or .cif file"""
    if pathlib.Path(filename).suffix.lower() == ".cif":
        parser = Bio.PDB.MMCIFParser(QUIET=quiet)
    else:
        parser = Bio.PDB.PDBParser(QUIET=quiet)

    struct = parser.get_structure("foo", filename)

    atoms = [a for a in struct.get_atoms()]
    natoms = len(atoms)
    coords = pyscatter.real_type(numpy.zeros((natoms, 3)))
    occupancy = pyscatter.real_type(numpy.zeros(natoms))
    elements = []

    if use_bfactor:
        bfactor = pyscatter.real_type(numpy.zeros(natoms))
    else:
        bfactor = None

    for i, a in enumerate(atoms):
        coords[i, :] = pyscatter.real_type(a.get_coord()*1e-10)  # Convert to m            
        elements.append(a.element.capitalize())
        occupancy[i] = a.occupancy
        if use_bfactor:
            bfactor[i] = a.bfactor * 1e-20  # Convert to m^2

    return AtomsSample(natoms, coords, elements, occupancy, bfactor)


def read_pdb_sloppy(filename: pathlib.Path, use_bfactor: bool=False
                    ) -> AtomsSample:
    """Reads all atoms of a .pdb file, regardless of if it makes sense or
    not"""
    use_bfactor = bool(use_bfactor)

    coords = []
    elements = []
    occupancy = []
    if use_bfactor:
        bfactor = []

    def parse_line(self, line):
        if (line[:4].upper() == "ATOM" or
                line[:6].upper() == "HETATM"):
            atom = {}
            atom["element"] = line[77:78+1].strip()
            atom["coord"] = (float(line[31:38+1].strip())*1e-10,
                             float(line[39:46+1].strip())*1e-10,
                             float(line[47:54+1].strip())*1e-10)
            atom["occupancy"] = float(line[55:60+1].strip())
            if self.use_bfactor:
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

    coords = pyscatter.real_type(numpy.array(coords))
    occupancy = pyscatter.real_type(numpy.array(occupancy))
    if use_bfactor:
        bfactor = pyscatter.real_type(numpy.array(bfactor))
    natoms = len(elements)
    return AtomsSample(natoms, coords, elements, occupancy, bfactor)
                

def calculate_fourier(sample: AtomsSample, detector: detector.Detector,
                      photon_energy: float,
                      rotation: Tuple[float, float, float, float] = (1, 0, 0, 0)
                      ) -> ArrayLike:
    if pyscatter.cupy_on():
        return calculate_fourier_cuda(
            sample, detector, photon_energy, rotation)
    else:
        return calculate_fourier_cpu(
            sample, detector, photon_energy, rotation)


def calculate_fourier_cpu(sample: AtomsSample, detector: detector.Detector,
                          photon_energy: float,
                          rotation: Tuple[float, float, float, float] = (1, 0, 0, 0)
                          ) -> ArrayLike:
    S = detector.scattering_vector(photon_energy, rotation)
    
    sf = StructureFactors()
    for element in sample.unique_elements():
        sf.precalculate_for_element(element, numpy.linalg.norm(S, axis=-1))
    
    diff = pyscatter.complex_type(numpy.zeros(detector.shape))

    use_bfactor = sample.bfactor is not None

    if use_bfactor:
        S_abs = numpy.linalg.norm(S, axis=-1)

    atom_iterator = zip(sample.coords, sample.occupancy, sample.elements)
    for coord, occupancy, element in atom_iterator:
        # coord_slice = (slice(None), ) + (None, )*len(S.shape[:-1])
        # dotp = (coord[coord_slice] * S).sum(axis=0)
        # dotp = (coord * S).sum(axis=-1)
        dotp = S @ coord

        atom_diff = (sf.precalculated[element] * occupancy *
                     numpy.exp(2j * numpy.pi * dotp))

        if use_bfactor:
            atom_diff *= numpy.exp(-sample.bfactor * S_abs**2 * 0.25)

        diff += atom_diff
    return diff


if pyscatter.cupy_on():
    __kernels = pyscatter.cuda_tools.import_cuda_file(
        'atoms_cuda.cu', ['calculate_scattering'])

    def calculate_scattering(element_diff: ArrayLike,
                             S: ArrayLike,
                             element_coords: ArrayLike,
                             element_occupancy: ArrayLike,
                             bfactor: Optional[ArrayLike]=None) -> None:
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
                               detector: detector.Detector,
                               photon_energy: float,
                               rotation: Tuple[float, float, float, float] = (1, 0, 0, 0)
                               ) -> ArrayLike:
        import cupy
        S = detector.scattering_vector(photon_energy, rotation)

        sf = StructureFactors()
        for element in sample.unique_elements():
            sf.precalculate_for_element(element, numpy.linalg.norm(S, axis=-1))

        diff = pyscatter.complex_type(numpy.zeros(detector.shape))

        unique_elements = sample.unique_elements()
        for element in unique_elements:
            element_array = pyscatter.always_numpy.array(sample.elements)
            selection = numpy.asarray(element_array == element,
                                      dtype="bool")
            element_coords = cupy.ascontiguousarray(
                sample.coords[selection], dtype="float32")
            element_occupancy = cupy.ascontiguousarray(
                sample.occupancy[selection], dtype="float32")
            
            use_bfactor = sample.bfactor is not None

            if use_bfactor:
                element_bfactor = cupy.ascontiguousarray(
                    sample.bfactor[selection], dtype="float32")
            else:
                element_bfactor = None

            element_diff = numpy.zeros_like(diff)
            pyscatter.cuda_extensions.calculate_scattering(
                element_diff, S, element_coords, element_occupancy,
                bfactor=element_bfactor)
            diff += sf.precalculated[element] * element_diff
        return diff