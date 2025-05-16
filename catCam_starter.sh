#!/usr/bin/env bash

echo "Executing CatPreyAnalyzer"
# Tensorflow Stuff
export PYTHONPATH=$PYTHONPATH:/home/pi/tensorflow/models/research:/home/pi/tensorflow/models/research/slim:/home/pi/.local/lib/python3.7/site-packages
cd /home/pi/CatPreyAnalyzer
python3 cascade.py $@
