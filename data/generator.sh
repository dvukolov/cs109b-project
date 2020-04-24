#!/bin/sh

set -e

python generator.py --size 200_000 --sersics 1000 --psf 0.5 --noise 200 data_v1.npz
python generator.py --size 200_000 --sersics 1000 --psf 0.5 data_v2.npz
