from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow>=1.2.0',
'gym>=0.1',
'seaborn>=0.1',
'pandas>=0.1',
'matplotlib>=0.1',
'numpy>=0.1',
'scipy>=0.1',
'opencv-python>=0.1']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My trainer application package.'
)