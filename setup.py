from setuptools import setup

setup(name="PyScatter",
      version="0.1",
      description="Tools for X-ray scattering simulations",
      author="Tomas Ekeberg",
      packages=["PyScatter"],
      package_data={"PyScatter": ["PyScatter/structure_factors.p"]})
