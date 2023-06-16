import PyScatter
import condor
from eke import rotmodule
from eke import time_tools
from eke import conversions

import numpy


pdbfile = "/Users/ekeberg/Work/Scratch/4o01.pdb"
# pdbfile = "/Users/ekeberg/Work/My Papers/GroEL/Figures/PDBfiles/1SS8_H2O_35_cyl.pdb"

# photon_energy = 1200
photon_energy = 3000
pulse_energy = 4e-3
focus_diameter = 0.1e-6

detector_distance = 0.15
pixel_size = 8*75e-6
pattern_shape = (128, 128)

# rotation = numpy.array([ 0.20109424, -0.86721112,  0.        ,  0.45552825])
rot = rotmodule.random()

w = time_tools.StopWatch()
w.start()
# pdb = PyScatter.SimplePDB(pdbfile)
pdb = PyScatter.SloppyPDB(pdbfile)
src = PyScatter.Source(photon_energy, pulse_energy, focus_diameter)
det = PyScatter.RectangularDetector(pattern_shape, pixel_size, detector_distance)


pattern = PyScatter.calculate_pattern(pdb, det, src, rot)

w.stop()

print(f"PyScatter took {w.time()} s")




w.start()
source = condor.Source(wavelength=conversions.ev_to_m(photon_energy), # eV
                       pulse_energy=pulse_energy, # J
                       focus_diameter=focus_diameter) # m

detector = condor.Detector(distance=detector_distance,
                           pixel_size=pixel_size,
                           nx=pattern_shape[0], ny=pattern_shape[1], # Detector size in pixels
                           cx=pattern_shape[0]/2-0.5, cy=pattern_shape[1]/2-0.5) # Detector center in pixels

particle = condor.ParticleAtoms(pdb_filename=pdbfile,
                                rotation_values=rotmodule.inverse(rot),
                                rotation_formalism="quaternion")

experiment = condor.Experiment(source, {"particle_atoms": particle}, detector)
result = experiment.propagate()

condor_pattern = result["entry_1"]["data_1"]["data"]
w.stop()

print(f"Condor took {w.time()} s")

vmax = max(pattern.max(), condor_pattern.max())
vmin = vmax * 1e-7

def plot_results():
    import matplotlib.pyplot
    fig = matplotlib.pyplot.figure("Compare")
    fig.clear()
    ax = fig.subplots(1, 2)
    p0 = ax[0].imshow(pattern, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    fig.colorbar(p0)
    p1 = ax[1].imshow(condor_pattern, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    fig.colorbar(p1)
