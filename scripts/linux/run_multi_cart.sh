#!/bin/sh

export DISPLAY=localhost:0.0

venv/bin/python3 src/main.py --config=local_qmix,qmix --env-config=multi_cart
venv/bin/python3 src/plot.py --config=local_qmix,qmix --env-config=multi_cart

