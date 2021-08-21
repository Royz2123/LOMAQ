#!/bin/sh

export DISPLAY=localhost:0.0
LANG=en_US

#xvfb-run -e /dev/stdout python3 src/main.py --config=qmix --env-config=multi_particle
#xvfb-run -e /dev/stdout python3 src/plot.py --config=qmix --env-config=multi_particle

python3 src/main.py --config=local_qmix --env-config=multi_particle
python3 src/plot.py --config=local_qmix --env-config=multi_cart


