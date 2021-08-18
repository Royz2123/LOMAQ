#!/bin/sh

export DISPLAY=localhost:0.0

xvfb-run -s \"-screen 0 1400x900x24\" python3 src/main.py --config=qmix --env-config=multi_particle
xvfb-run -s \"-screen 0 1400x900x24\" python3 src/plot.py --config=qmix --env-config=multi_particle
#python3 src/plot.py --config=iql_local,local_qmix,qmix --env-config=multi_cart

