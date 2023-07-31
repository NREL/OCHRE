import io
import os
import re

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

requirements = [
    'numpy',
    'pandas>=1.4.0',
    'scipy',
    'matplotlib',
    'pint',
    'pvlib',
    'NREL-PySAM',
    'psychrolib',
    'numba',  # required for psychrolib
    'xmltodict',
    'pyarrow',
    'fastparquet',
    'rainflow',
    'pytz',
    'python-dateutil',
    # 'h5py',
]


# Read the version from the __init__.py file without importing it
def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(name='ochre',
      version=find_version('ochre', '__init__.py'),
      description='A residential energy model with controllable equipment and DERs for building-to-grid co-simulation',
      author='Jeff Maguire',
      author_email='Jeff.Maguire@nrel.gov',
      maintainer='Michael Blonsky',
      maintainer_email='Michael.Blonsky@nrel.gov',
      url='https://github.com/NREL/OCHRE',
      packages=['ochre', 'ochre.utils', 'ochre.Equipment', 'ochre.Models'],
      python_requires='>=3.9',
      install_requires=requirements,
      package_data={'ochre': ['../defaults/*', '../defaults/*/*']},
      )
