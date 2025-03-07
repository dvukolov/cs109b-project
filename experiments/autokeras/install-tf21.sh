#!/bin/sh

# Autokeras requires TensorFlow 2.1 or higher. Install it into a separate environment.

conda create --name tf21 -y python=3.7 ipykernel
conda activate tf21

python -m ipykernel install --user --name tf21 --display-name "Python (TF2.1)"
# The output of the following command should list the tf21 environment:
# jupyter kernelspec list

# General dependencies
conda install -c anaconda ipywidgets matplotlib numpy pandas scikit-learn seaborn tensorflow-gpu tqdm
pip install tensorflow-addons hiplot

# Autokeras-specific dependencies
conda install -c conda-forge packaging pyparsing pytz keras-tuner typeguard
pip install autokeras

# To remove the environment run:
# conda remove --name tf21 --all
# jupyter kernelspec uninstall tf21
