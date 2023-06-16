import numpy
import PyScatter
import nfft
from eke import elements
from eke import tools
from eke import rotmodule
from eke import conversions
from eke import constants
from eke import time_tools

photon_energy = 2000
pulse_energy = 4e-3
focus_diameter = 0.1e-6

detector_distance = 0.15
pixel_size = 8*75e-6
pattern_shape = (128, 128)

map_shape = (64, )*3
particle_size = 20e-9

material = elements.MATERIALS["protein"]
density_map = numpy.roll(tools.round_mask(map_shape, 10), 10, axis=0) + numpy.roll(tools.round_mask(map_shape, 10), -10, axis=0)

w = time_tools.StopWatch()
w.start()

det = PyScatter.RectangularDetector(pattern_shape, pixel_size, detector_distance)
src = PyScatter.Source(photon_energy, pulse_energy, focus_diameter)
rot = rotmodule.random()
# rot = rotmodule.from_angle_and_dir(0.9, (1, 1, 1))

sample = PyScatter.MapSample(particle_size/map_shape[0], density_map, material)
pattern = PyScatter.calculate_pattern(sample, det, src, rot)

w.stop()
print(f"PyScatter took {w.time()} s")


import condor

w.start()

source = condor.Source(wavelength=conversions.ev_to_m(photon_energy),
                       pulse_energy=pulse_energy,
                       focus_diameter=focus_diameter)

detector = condor.Detector(distance=detector_distance,
                           pixel_size=pixel_size,
                           nx=pattern_shape[0], ny=pattern_shape[1],
                           cx=pattern_shape[0]/2-0.5, cy=pattern_shape[1]/2-0.5)

condor_map = numpy.bool_(density_map).swapaxes(0, 2)

particle = condor.ParticleMap(map3d=condor_map,
                              dx=particle_size/map_shape[0],
                              geometry="custom", 
                              material_type="custom",
                              massdensity=material.material_density(),
                              atomic_composition=material.element_ratios(),
                              rotation_values=rotmodule.inverse(rot),
                              rotation_formalism="quaternion")


experiment = condor.Experiment(source, {"particle_map": particle}, detector)
result = experiment.propagate()

condor_pattern = result["entry_1"]["data_1"]["data"]

w.stop()
print(f"Condor took {w.time()} s")

def plot_results():
    import matplotlib.pyplot
    vmax = max(pattern.max(), condor_pattern.max())
    vmin = vmax * 1e-7

    fig = matplotlib.pyplot.figure("Compare")
    fig.clear()
    ax = fig.subplots(1, 2)
    p0 = ax[0].imshow(pattern, norm=matplotlib.colors.LogNorm())#vmin=vmin, vmax=vmax))
    fig.colorbar(p0)
    p1 = ax[1].imshow(condor_pattern, norm=matplotlib.colors.LogNorm())#vmin=vmin, vmax=vmax))
    fig.colorbar(p1)


# Check
# What is dx in condor. Most likely the pixel size in real space.
# Do we get the map (prior to Fourier transform) in the same way
