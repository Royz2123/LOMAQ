#!/bin/sh

export DISPLAY=localhost:0.0

venv/bin/python3 src/multi_main.py --config=lomaq,qmix --env-config=multi_cart

