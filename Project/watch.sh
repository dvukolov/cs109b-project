#!/bin/sh

# Update the slides whenever the notebooks change.
# Requires `entr`, available for Linux and Mac.

ls presentation*.ipynb | entr -r ./slides.sh 

