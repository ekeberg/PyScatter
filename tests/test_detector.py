import pytest
import numpy
from PyScatter import detector


def test_quaternion_to_matrix():
    q = numpy.array([numpy.sqrt(2)/2, numpy.sqrt(2)/2, 0, 0])  # 90 degree rotation around x
    expected = numpy.array([[1, 0, 0],
                            [0, 0, -1],
                            [0, 1, 0]])
    m = detector.quaternion_to_matrix(q)
    assert numpy.allclose(m, expected)


def test_rotate():
    q = numpy.array([numpy.sqrt(2)/2, numpy.sqrt(2)/2, 0, 0])  # 90 degree rotation around x    
    coordinates = numpy.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
    expected = numpy.array([[1, 0, 0],
                            [0, 0, 1],
                            [0, -1, 0]])
    rotated = detector.rotate(q, coordinates)
    assert numpy.allclose(rotated, expected)


def test_scattering_vector():
    photon_energy = 3000
    detector_distance = 0.15
    pixel_size = 0.01
    det = detector.RectangularDetector((4, 4), pixel_size,
                                       detector_distance,
                                       center=(0, 0))
    s = det.scattering_vector(photon_energy)

    wavelength = detector.conversions.ev_to_m(photon_energy)
    
    expected00 = numpy.array([0, 0, 0])

    s0 = 1/wavelength * numpy.array([0, 0, 1])
    s1 = numpy.array([pixel_size*3, 0, detector_distance])
    s1 = s1 / numpy.linalg.norm(s1) / wavelength
    expected30 = s1-s0

    assert len(s.shape) == 3

    assert numpy.allclose(s[0, 0], expected00)
    # 0, 3 since the x axis is vertical and the y axis is horizontal
    assert numpy.allclose(s[0, 3], expected30)
    

def test_scattering_angle():
    photon_energy = 3000
    detector_distance = 0.15
    pixel_size = 0.01
    det = detector.RectangularDetector((4, 4), pixel_size,
                                       detector_distance,
                                       center=(0, 0))
    angle = det.scattering_angle()

    assert len(angle.shape) == 2

    expected = numpy.arctan(numpy.sqrt(2*(pixel_size*3)**2)/detector_distance)
    assert numpy.allclose(angle[0, 0], 0)
    assert numpy.allclose(angle[3, 3], expected)


def test_solid_angle_large():
    """Check that the solid angle of a very large pixel is 2pi"""
    x = numpy.array([0])
    y = numpy.array([0])
    pixel_size = (1e40, 1e40)
    detector_distance = 0.15
    solid_angle = detector.solid_angle_rectangular_pixels(
        x, y, pixel_size, detector_distance)
    assert numpy.allclose(solid_angle, 2*numpy.pi)


def test_fourier_scattering_angle():
    max_scattering_angle = 0.2
    det = detector.FourierDetector((5, 5, 5), max_scattering_angle)
    angle = det.scattering_angle()

    # Check that the edge angle is actually the specified max angle
    assert angle[0, 2, 2] == pytest.approx(max_scattering_angle)


def test_fourier_scattering_vector():
    max_scattering_angle = 0.2
    photon_energy = 3000
    det = detector.FourierDetector((5, 5, 5), max_scattering_angle)
    s = det.scattering_vector(photon_energy)

    
