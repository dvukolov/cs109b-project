#!/bin/bash

./gen_test.py --size 10_000 --noise=400 snr30.npz
./gen_test.py --size 10_000 --noise=200 snr60.npz
