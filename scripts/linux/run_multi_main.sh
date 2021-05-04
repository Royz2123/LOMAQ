#!/bin/sh

export DISPLAY=localhost:0.0

"venv\Scripts\python.exe" src/multi_main.py --config=local_qmix,qmix --env-config=multi_cart"venv\Scripts\python.exe" src/plot.py --config=local_qmix,qmix --env-config=multi_cart

