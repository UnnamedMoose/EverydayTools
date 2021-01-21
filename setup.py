""" A setuptools-based setup module for my EverydayTools.
"""
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
import os

# Get the long description from the README file
with open(os.path.join(os.getcwd(), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get the version from a file, which is included with the distribution (listed in MANIFEST.in).
with open(os.path.join(os.getcwd(), 'VERSION'), encoding='utf-8') as version_file:
    ver=version_file.read().strip()

setup(
    name='everydaytools',
    version=ver, # Use the version from the file.

    zip_safe= False,

    description='Tools for doing almost anything.',
    long_description=long_description, # From the README.md

    # The project's main homepage.
    url='https://github.com/UnnamedMoose/EverydayTools',

    # Author details
    author='Artur Lidtke',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: enthusiasts',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
    ],

    keywords='magic, dwarves, data',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=['pandas>=0.19.2','matplotlib>=3.0.3','numpy>=1.12.1'],

    extras_require={
        'dev': ['pdb'],
        'test': ['unittest']
    },

    package_data={
    # TODO could add some package data for testing.
	    # Include .tle files from any package.
        #'': ['*.tle'],
        # Also include a specific files.
        #'TLEFiltering': ['10983.tle'],
    },

    # Automatically install the entry-point scripts.
    # TODO would be good to create entry-point scripts if there will be any.
    #entry_points={
    #    'console_scripts': [
    #        'serialMonitor=SerialMonitor:main',
    #    ],
    #},
)
