from setuptools import setup, find_packages


setup(
    name='xxneuralnet',
    version='0.1',
    package_data={'': ['README.md']},
    description='Fully customizable neural network',
    author='Blake Williams',
    author_email='blakerichardwills@gmail.com',
    packages=find_packages(),
    install_requires=['pandas']
)
