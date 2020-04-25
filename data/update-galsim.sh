#!/bin/bash

# Install GalSim with an increased Sersic index cache of 1000

set -e

# Remove GalSim
conda uninstall --force -y galsim

# Install dependencies
sudo apt-get -qq update
sudo apt-get install -y python3-dev libfftw3-dev libeigen3-dev pybind11-dev

# Get the source code with an increased cache size of 1000
git clone --single-branch --branch sersic-cache https://github.com/dvukolov/GalSim.git

# Build and install GalSim
pushd GalSim
python setup.py install
popd

# Check installation
python -c "import galsim" && echo "GalSim installed successfully"
