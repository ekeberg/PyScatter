import pickle
import Bio.PDB
import pathlib
# from ..pyscatter import *
from .. import pyscatter
numpy = pyscatter.numpy


import importlib.resources as pkg_resources
STRUCTURE_FACTOR_TABLE = pickle.loads(
    pkg_resources.read_binary("PyScatter", "structure_factors.p"))


# STRUCTURE_FACTOR_TABLE = pickle.load(open(
#     pathlib.Path(__file__).parent / "structure_factors.p", "rb"))
                                          

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
        self.precalculated[element] = pyscatter.real_type(
            self.structure_factor(element, S_array))


class AbstractPDB:
    def __init__(self, filename):
        pass

    def unique_elements(self):
        # return numpy.unique(self.elements)
        return list(set(self.elements))


class SimplePDB(AbstractPDB):
    """Uses the Biopython parser"""
    def __init__(self, filename, bfactor=False):
        if pathlib.Path(filename).suffix.lower() == ".cif":
            self._parser = Bio.PDB.MMCIFParser(QUIET=True)
        else:
            self._parser = Bio.PDB.PDBParser(QUIET=True)
        self.struct = self._parser.get_structure("foo", filename)

        self.use_bfactor = bool(bfactor)

        atoms = [a for a in self.struct.get_atoms()]
        self.natoms = len(atoms)
        # self.coords = real_type(numpy.zeros((self.natoms, 3)))
        self.coords = pyscatter.always_numpy.zeros((self.natoms, 3))
        self.elements = []
        # self.occupancy = real_type(numpy.zeros(self.natoms))
        self.occupancy = pyscatter.always_numpy.zeros(self.natoms)

        if self.use_bfactor:
            self.bfactor = pyscatter.always_numpy.zeros(self.natoms)

        for i, a in enumerate(atoms):
            self.coords[i, :] = a.get_coord()*1e-10  # Convert to m            
            self.elements.append(a.element.capitalize())
            self.occupancy[i] = a.occupancy
            if self.use_bfactor:
                self.bfactor[i] = a.bfactor * 1e-20  # Convert to m^2

        self.coords = pyscatter.real_type(self.coords)
        self.occupancy = pyscatter.real_type(self.occupancy)


class SloppyPDB(AbstractPDB):
    """Reads all atoms, regardless of if it makes sense or not"""
    def __init__(self, filename, bfactor=False):

        self.use_bfactor = bool(bfactor)

        self.coords = []
        self.elements = []
        self.occupancy = []
        if self.use_bfactor:
            self.bfactor = []
        
        with open(filename) as f:
            for line in f.readlines():
                atom = self._parse_line(line)

                if atom is not None:
                    self.coords.append(atom["coord"])
                    self.elements.append(atom["element"])
                    self.occupancy.append(atom["occupancy"])
                    if self.use_bfactor:
                        self.bfactor.append(atom["bfactor"])

        self.coords = pyscatter.real_type(numpy.array(self.coords))
        self.occupancy = pyscatter.real_type(numpy.array(self.occupancy))
        if self.use_bfactor:
            self.bfactor = pyscatter.real_type(numpy.array(self.bfactor))
        self.natoms = len(self.elements)
                    
    def _parse_line(self, line):
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


def calculate_fourier_from_pdb(pdb, detector, photon_energy,
                               rotation=(1, 0, 0, 0)):
    if pyscatter.cupy_on():
        return calculate_fourier_from_pdb_cuda(
            pdb, detector, photon_energy, rotation)
    else:
        return calculate_fourier_from_pdb_cpu(
            pdb, detector, photon_energy, rotation)


def calculate_fourier_from_pdb_cpu(pdb, detector, photon_energy,
                                   rotation=(1, 0, 0, 0)):
    S = detector.scattering_vector(photon_energy, rotation)
    
    sf = StructureFactors()
    for element in pdb.unique_elements():
        sf.precalculate_for_element(element, numpy.linalg.norm(S, axis=-1))
    
    diff = pyscatter.complex_type(numpy.zeros(detector.shape))

    if pdb.use_bfactor:
        S_abs = numpy.linalg.norm(S, axis=-1)

    atom_iterator = zip(pdb.coords, pdb.occupancy, pdb.elements)
    for coord, occupancy, element in atom_iterator:
        # coord_slice = (slice(None), ) + (None, )*len(S.shape[:-1])
        # dotp = (coord[coord_slice] * S).sum(axis=0)
        # dotp = (coord * S).sum(axis=-1)
        dotp = S @ coord

        atom_diff = (sf.precalculated[element] * occupancy *
                     numpy.exp(2j * numpy.pi * dotp))

        if pdb.use_bfactor:
            atom_diff *= numpy.exp(-pdb.bfactor * S_abs**2 * 0.25)

        diff += atom_diff
    return diff


def calculate_fourier_from_pdb_cuda(pdb, detector, photon_energy,
                                    rotation=(1, 0, 0, 0)):
    import cupy
    S = detector.scattering_vector(photon_energy, rotation)

    sf = StructureFactors()
    for element in pdb.unique_elements():
        sf.precalculate_for_element(element, numpy.linalg.norm(S, axis=-1))

    diff = pyscatter.complex_type(numpy.zeros(detector.shape))

    unique_elements = list(pyscatter.always_numpy.unique(pdb.elements))
    for element in unique_elements:
        selection = numpy.asarray(pyscatter.always_numpy.array(pdb.elements) == element,
                                  dtype="bool")
        element_coords = cupy.ascontiguousarray(pdb.coords[selection],
                                                dtype="float32")
        element_occupancy = cupy.ascontiguousarray(pdb.occupancy[selection],
                                                   dtype="float32")
        
        if pdb.use_bfactor:
            element_bfactor = cupy.ascontiguousarray(
                pdb.bfactor[selection], dtype="float32")
        else:
            element_bfactor = None

        element_diff = numpy.zeros_like(diff)
        pyscatter.cuda_extensions.calculate_scattering(
            element_diff, S, element_coords, element_occupancy,
            bfactor=element_bfactor)
        diff += sf.precalculated[element] * element_diff
    return diff


def calculate_pattern_from_pdb(pdb, detector, source, rotation=(1, 0, 0, 0)):
    diff = calculate_fourier_from_pdb(
        pdb, detector, source.photon_energy, rotation)
    pattern = fourier_to_pattern(diff, detector, source)
    return pattern
