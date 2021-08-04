#!/bin/sh

export DISPLAY=localhost:0.0

python3 src/main.py --config=local_qmix --env-config=multi_cart
python3 src/plot.py --config=local_qmix --env-config=multi_cart
#python3 src/plot.py --config=iql_local,local_qmix,qmix --env-config=multi_cart

